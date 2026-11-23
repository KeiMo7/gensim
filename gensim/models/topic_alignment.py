#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Stable, content-derived topic identifiers across successive training batches.

Why
---
LDA topic indices are positional: after retraining on a fresh batch, topic ``7`` may
describe an entirely different subject. Reports that refer to topics by number then
have to be rewritten from scratch. This module matches topics by **content** (their
probability distribution over words), assigns identifiers from that match, and stores
the result in a durable ledger so later batches — in any process, on any machine —
receive the same numbers.

Guarantees
----------
* Identifiers derive only from content. Ties are broken by term id and local index,
  never by training order or by the random initialisation order.
* The alignment result, the drift scores and the document-topic remapping are all
  computed from the same :class:`AlignmentResult`; there is no second calculation
  that could disagree.
* Every previous topic ends a batch with an explicit fate: ``matched``, ``split``
  (the parent identifier follows the branch with the larger share of the old topic's
  mass, the other branch gets a new number), ``merged`` (the secondary number records
  its destination) or ``disappeared`` (with a destination for its document mass, never
  silently dropped).
* Drift is reported as ``"not_comparable"`` when the two signatures share too few
  words — never as a misleading zero.
* Ledger writes are atomic. A failed write raises :class:`LedgerWriteError` and rolls
  the in-memory ledger back, so a caller cannot mistake a failed commit for success.
* Fully opt-in: :class:`AlignmentConfig` defaults to ``enabled=False``; model training
  and topic output are untouched when the feature is off.

Example
-------
.. sourcecode:: pycon

    >>> from gensim.models.topic_alignment import AlignmentConfig, TopicAligner
    >>> cfg = AlignmentConfig(enabled=True, ledger_path='topics-ledger.json')
    >>> result = TopicAligner(cfg).align_and_commit(model, batch_id='2026-11')
    >>> result.summary()['matched']
"""

import hashlib
import json
import logging
import math
import os
import tempfile
import time
from collections import defaultdict

import numpy as np


logger = logging.getLogger(__name__)

LEDGER_FORMAT = "gensim-topic-ledger"
LEDGER_VERSION = 1

#: Prefix of a brand-new identifier, e.g. ``"T0007"``.
UID_PREFIX = "T"

#: Per-topic fate tags, used in ledger history and reports.
STATUS_MATCHED = "matched"
STATUS_BORN = "born"
STATUS_MERGED = "merged"
STATUS_SPLIT = "split"
STATUS_DISAPPEARED = "disappeared"
STATUS_RETIRED = "retired"
STATUS_ABSORBED = "absorbed"

#: Drift-score sentinel used when two topics cannot be compared.
DRIFT_NOT_COMPARABLE = "not_comparable"

#: Drift status when a score is computable but above the configured threshold.
DRIFT_THRESHOLD_EXCEEDED = "threshold_exceeded"


class AlignmentError(RuntimeError):
    """Raised when alignment cannot be performed without producing misleading output."""


class LedgerError(AlignmentError):
    """Raised when the ledger is missing, corrupt, or inconsistent with the current batch."""


class LedgerWriteError(AlignmentError):
    """Raised when a ledger write fails; callers must not report training success then."""


def _atomic_write_json(path, payload):
    """Write ``payload`` to ``path`` atomically; raise :class:`LedgerWriteError` on failure."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    tmp_path = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".ledger-", suffix=".tmp", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError as err:
        raise LedgerWriteError("could not write ledger %r: %s" % (path, err)) from err
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("could not remove temporary ledger file %r", tmp_path)


def _fingerprint(payload):
    """Stable content fingerprint of a JSON payload, for cross-process comparison."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AlignmentConfig:
    """Configuration for cross-batch topic alignment.

    Parameters
    ----------
    enabled : bool, optional
        Master switch. ``False`` (default) means no alignment code runs at all.
    ledger_path : str, optional
        Path of the JSON ledger. Required when ``enabled`` is True and no explicit
        :class:`TopicLedger` is supplied to :class:`TopicAligner`.
    similarity_threshold : float, optional
        Minimum cosine similarity between two topic signatures for a match. ``[0, 1]``.
    drift_threshold : float, optional
        Drift score above which a matched topic is flagged ``"threshold_exceeded"``.
        Computable scores lie in ``[0, 1]``. Flagging never changes the match itself.
    min_shared_terms : int, optional
        Minimum shared words for a drift score to be defined; below it the score is
        reported as ``"not_comparable"``.
    topn : int, optional
        Number of most probable words used to build a topic signature.
    max_topics : int, optional
        Cap on the number of active identities per batch. Excess topics are absorbed
        into the most similar surviving topic and recorded. ``None`` means no cap.
    retired_grace : int, optional
        Number of consecutive batches a disappeared topic stays dormant before it is
        marked retired. ``0`` retires it immediately.
    require_previous : bool, optional
        When True, aligning through a ledger with no recorded batch raises
        :class:`LedgerError` instead of silently starting a new history. Useful for
        scheduled reports where a missing ledger is a real fault.

    """

    def __init__(self, enabled=False, ledger_path=None, similarity_threshold=0.35,
                 drift_threshold=0.55, min_shared_terms=5, topn=30, max_topics=None,
                 retired_grace=0, require_previous=False):
        self.enabled = bool(enabled)
        self.ledger_path = ledger_path
        self.similarity_threshold = float(similarity_threshold)
        self.drift_threshold = float(drift_threshold)
        self.min_shared_terms = int(min_shared_terms)
        self.topn = int(topn)
        self.max_topics = None if max_topics is None else int(max_topics)
        self.retired_grace = int(retired_grace)
        self.require_previous = bool(require_previous)

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1], got %r" % (self.similarity_threshold,))
        if not 0.0 <= self.drift_threshold <= 1.0:
            raise ValueError("drift_threshold must be in [0, 1], got %r" % (self.drift_threshold,))
        if self.min_shared_terms < 1:
            raise ValueError("min_shared_terms must be >= 1, got %r" % (self.min_shared_terms,))
        if self.topn < 1:
            raise ValueError("topn must be >= 1, got %r" % (self.topn,))
        if self.max_topics is not None and self.max_topics < 1:
            raise ValueError("max_topics must be >= 1 or None, got %r" % (self.max_topics,))
        if self.retired_grace < 0:
            raise ValueError("retired_grace must be >= 0, got %r" % (self.retired_grace,))

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "enabled": self.enabled,
            "ledger_path": self.ledger_path,
            "similarity_threshold": self.similarity_threshold,
            "drift_threshold": self.drift_threshold,
            "min_shared_terms": self.min_shared_terms,
            "topn": self.topn,
            "max_topics": self.max_topics,
            "retired_grace": self.retired_grace,
            "require_previous": self.require_previous,
        }

    @classmethod
    def from_dict(cls, payload):
        """Build a configuration from :meth:`to_dict` output (unknown keys ignored)."""
        known = set(cls.__init__.__code__.co_varnames) - {"self"}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def __repr__(self):
        return "AlignmentConfig(%s)" % ", ".join(
            "%s=%r" % (k, v) for k, v in sorted(self.to_dict().items()))


class TopicSignature:
    """A content fingerprint for one topic: a normalised distribution over top words.

    Parameters
    ----------
    words : list of (str, float)
        Word/probability pairs, most probable first.
    batch_id : str
        Batch the topic was trained on.
    local_index : int
        Positional index inside its own model (``0 .. num_topics - 1``).

    """

    def __init__(self, words, batch_id, local_index):
        total = sum(p for _, p in words)
        if total <= 0:
            raise ValueError("a topic signature needs at least one positive weight")
        self.words = [(str(w), float(p) / total) for w, p in words]
        self.batch_id = str(batch_id)
        self.local_index = int(local_index)
        self._lookup = dict(self.words)
        self._norm = math.sqrt(sum(p * p for _, p in self.words))

    @property
    def terms(self):
        """Set of words in this signature."""
        return set(self._lookup)

    def cosine(self, other):
        """Cosine similarity against another signature; ``0.0`` when no words are shared."""
        shared = self.terms & other.terms
        if not shared:
            return 0.0
        dot = sum(self._lookup[w] * other._lookup[w] for w in shared)
        denominator = self._norm * other._norm
        return float(dot / denominator) if denominator else 0.0

    def shared_info(self, other):
        """Return ``(cosine, n_shared, coverage_self, coverage_other)``.

        Coverage is the fraction of a signature's probability mass that lies on words
        shared with the other signature. It answers "how much of this topic does the
        other account for?" and is the evidence used to choose a split's primary branch.
        """
        shared = self.terms & other.terms
        if not shared:
            return 0.0, 0, 0.0, 0.0
        return (
            self.cosine(other),
            len(shared),
            float(sum(self._lookup[w] for w in shared)),
            float(sum(other._lookup[w] for w in shared)),
        )

    def to_dict(self):
        """Return a JSON-friendly representation (rounded to keep the ledger compact)."""
        return {
            "batch_id": self.batch_id,
            "local_index": self.local_index,
            "words": [[w, round(p, 8)] for w, p in self.words],
        }

    @classmethod
    def from_dict(cls, payload):
        """Inverse of :meth:`to_dict`."""
        return cls([(w, p) for w, p in payload["words"]],
                   batch_id=payload["batch_id"], local_index=payload["local_index"])

    def __repr__(self):
        return "TopicSignature(%s#%d: %s)" % (
            self.batch_id, self.local_index, ", ".join(w for w, _ in self.words[:5]))


def _js_distance(a, b):
    """Jensen-Shannon distance between two dense probability vectors."""
    m = 0.5 * (a + b)
    left = np.sum(np.where(a > 0, a * np.log(np.where(a > 0, a, 1.0) / np.where(m > 0, m, 1.0)), 0.0))
    right = np.sum(np.where(b > 0, b * np.log(np.where(b > 0, b, 1.0) / np.where(m > 0, m, 1.0)), 0.0))
    return float(math.sqrt(max(0.5 * (left + right), 0.0)))


def drift_score(previous, current, min_shared_terms=5):
    """Compute a ``[0, 1]`` drift score for two signatures of the same topic.

    The score is the Jensen-Shannon distance restricted to the shared vocabulary,
    divided by ``sqrt(2)`` (its maximum), so ``0`` means unchanged and ``1`` means
    maximally changed.

    Parameters
    ----------
    previous, current : TopicSignature
    min_shared_terms : int, optional
        Minimum shared vocabulary. With fewer shared words the distributions are not
        comparable and the function returns ``(None, "not_comparable")`` rather than a
        misleading zero.

    Returns
    -------
    tuple of (float or None, str)

    """
    shared = previous.terms & current.terms
    if len(shared) < min_shared_terms:
        return None, DRIFT_NOT_COMPARABLE
    vocabulary = sorted(shared)
    a = np.array([previous._lookup[w] for w in vocabulary], dtype=np.float64)
    b = np.array([current._lookup[w] for w in vocabulary], dtype=np.float64)
    a /= a.sum()
    b /= b.sum()
    return float(min(max(_js_distance(a, b) / math.sqrt(2.0), 0.0), 1.0)), "ok"


class TopicIdentity:
    """A stable identifier plus the ledger-side state needed to match it next batch.

    Parameters
    ----------
    topic_uid : str
        Stable identifier, e.g. ``"T0007"``.
    label : str
        Human-readable label derived from the top words at birth.
    origin_batch : str
        Batch in which the identifier was first created.
    local_index : int or None
        Position in the most recent batch; ``None`` when dormant/merged/absorbed.
    revision : int
        Number of batches in which the identifier has been active.
    missing : int
        Consecutive batches in which the identifier had no successor.
    history : list of dict
        Append-only audit trail.
    signature : TopicSignature or None
        Content as of the most recent active batch.

    """

    def __init__(self, topic_uid, label, origin_batch, local_index=None, revision=0,
                 missing=0, history=None, signature=None):
        self.topic_uid = str(topic_uid)
        self.label = str(label)
        self.origin_batch = str(origin_batch)
        self.local_index = None if local_index is None else int(local_index)
        self.revision = int(revision)
        self.missing = int(missing)
        self.history = list(history or [])
        self.signature = signature

    @property
    def active(self):
        """Whether this identity occupies a topic slot in the latest batch."""
        return self.local_index is not None

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "topic_uid": self.topic_uid,
            "label": self.label,
            "origin_batch": self.origin_batch,
            "local_index": self.local_index,
            "revision": self.revision,
            "missing": self.missing,
            "history": self.history,
            "signature": None if self.signature is None else self.signature.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload):
        """Inverse of :meth:`to_dict`."""
        signature = payload.get("signature")
        return cls(
            payload["topic_uid"], payload["label"], payload["origin_batch"],
            local_index=payload.get("local_index"), revision=payload.get("revision", 0),
            missing=payload.get("missing", 0), history=payload.get("history", []),
            signature=TopicSignature.from_dict(signature) if signature else None,
        )

    def __repr__(self):
        return "TopicIdentity(%s, %r)" % (self.topic_uid, self.label)


def _make_label(words, max_words=3):
    pieces = [w for w, _ in words[:max_words]]
    return " / ".join(pieces) if pieces else "unnamed"


class TopicMatch:
    """One accepted previous-topic -> current-topic correspondence with its evidence."""

    def __init__(self, old, new_local_index, similarity, n_shared_terms,
                 drift=None, drift_status="ok", evidence=None):
        self.old = old
        self.new_local_index = int(new_local_index)
        self.similarity = float(similarity)
        self.n_shared_terms = int(n_shared_terms)
        self.drift = drift
        self.drift_status = drift_status
        self.evidence = dict(evidence or {})

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "topic_uid": self.old.topic_uid,
            "label": self.old.label,
            "previous_local_index": self.old.local_index,
            "new_local_index": self.new_local_index,
            "similarity": round(self.similarity, 6),
            "shared_terms": self.n_shared_terms,
            "drift": None if self.drift is None else round(self.drift, 6),
            "drift_status": self.drift_status,
            "evidence": self.evidence,
        }


class AlignmentResult:
    """The single, self-consistent outcome of aligning one batch.

    Identifiers, drift scores, merge/split/birth/disappearance accounting and the
    document-topic remapping are all derived from this object, so they cannot disagree.

    Attributes
    ----------
    batch_id, previous_batch_id : str or None
    signature_map : dict
        ``{new local index: TopicSignature}`` for the aligned model.
    matches : list of TopicMatch
    births : list of TopicIdentity
    merges : list of dict
        ``{"survivor", "absorbed", "new_local_index", "similarity", "reason"}``.
    splits : list of dict
        ``{"parent", "primary_new_local_index", "secondary", "secondary_new_local_index",
        "parent_share", "secondary_share", "reason"}``.
    disappearances : list of dict
        Previous topics with no successor, with their document destination.
    retirements : list of dict
        Disappeared topics that exhausted their grace period.
    absorptions : list of dict
        Topics dropped to respect ``max_topics``.
    local_to_uid, previous_local_to_uid, document_remap : dict
    diagnostics : dict

    """

    def __init__(self, batch_id, previous_batch_id=None):
        self.batch_id = str(batch_id)
        self.previous_batch_id = None if previous_batch_id is None else str(previous_batch_id)
        self.signature_map = {}
        self.matches = []
        self.births = []
        self.merges = []
        self.splits = []
        self.disappearances = []
        self.retirements = []
        self.absorptions = []
        self.revivals = []
        self.local_to_uid = {}
        self.previous_local_to_uid = {}
        self.document_remap = {}
        self.diagnostics = {}

    @property
    def uid_to_local(self):
        """Inverse of :attr:`local_to_uid`."""
        return {uid: idx for idx, uid in self.local_to_uid.items()}

    def remap_document_topics(self, distribution):
        """Remap one document-topic distribution from previous to current local indices.

        Parameters
        ----------
        distribution : iterable of (int, float)
            ``(previous local topic index, weight)`` pairs, e.g. one row of
            ``model.get_document_topics(bow)`` for the previous batch.

        Returns
        -------
        tuple of (list of (int, float), dict)
            The remapped distribution and a conservation report. Mass is conserved:
            weights of disappeared/merged topics are folded into their recorded
            destination, and any mass with no destination is reported explicitly (it
            is never silently rescaled).

        """
        weights = defaultdict(float)
        for topic_id, weight in distribution:
            weights[int(topic_id)] += float(weight)

        input_mass = sum(weights.values())
        output = defaultdict(float)
        dropped = 0.0
        for old_index, weight in weights.items():
            destination = self.document_remap.get(old_index)
            if destination is None:
                dropped += weight
            else:
                output[int(destination)] += weight

        output_mass = sum(output.values())
        denominator = max(1e-12, abs(input_mass))
        conserved = abs((output_mass + dropped) - input_mass) <= 1e-9 * denominator
        report = {
            "input_mass": input_mass,
            "output_mass": output_mass,
            "dropped_mass": dropped,
            "total_conserved": bool(conserved),
        }
        return sorted(output.items()), report

    def summary(self):
        """Return a compact, JSON-friendly summary."""
        comparable = [m for m in self.matches if m.drift is not None]
        return {
            "batch_id": self.batch_id,
            "previous_batch_id": self.previous_batch_id,
            "num_topics": len(self.local_to_uid),
            "matched": len(self.matches),
            "born": len(self.births),
            "merged": len(self.merges),
            "split": len(self.splits),
            "disappeared": len(self.disappearances),
            "retired": len(self.retirements),
            "absorbed": len(self.absorptions),
            "revived": len(self.revivals),
            "drift_comparable": len(comparable),
            "drift_threshold_exceeded": sum(
                1 for m in self.matches if m.drift_status == DRIFT_THRESHOLD_EXCEEDED),
            "drift_not_comparable": sum(
                1 for m in self.matches if m.drift_status == DRIFT_NOT_COMPARABLE),
            "mean_drift": float(np.mean([m.drift for m in comparable])) if comparable else None,
            "max_drift": float(max(m.drift for m in comparable)) if comparable else None,
            "diagnostics": self.diagnostics,
        }

    def to_report(self):
        """Return a reviewable report: identifiers, words and counts only, no documents."""
        topics = []
        for index, uid in sorted(self.local_to_uid.items()):
            match = next((m for m in self.matches if m.old.topic_uid == uid), None)
            birth = next((b for b in self.births if b.topic_uid == uid), None)
            split = next((s for s in self.splits if s["secondary"] == uid), None)
            revival = next((r for r in self.revivals if r["topic_uid"] == uid), None)
            topics.append({
                "topic_uid": uid,
                "local_index": index,
                "previous_local_index": match.old.local_index if match else None,
                "label": (match.old.label if match else
                          (birth.label if birth else (revival["label"] if revival else ""))),
                "similarity": (round(match.similarity, 6) if match else
                               (revival["similarity"] if revival else None)),
                "drift": None if match is None or match.drift is None else round(match.drift, 6),
                "drift_status": ("new" if match is None and revival is None
                                 else ("revived" if revival is not None else match.drift_status)),
                "split_from": None if split is None else split["parent"],
                "revived": revival is not None,
                "top_words": [w for w, _ in self.signature_map[index].words[:10]]
                if index in self.signature_map else [],
            })
        return {
            "summary": self.summary(),
            "topics": topics,
            "matches": [m.to_dict() for m in self.matches],
            "births": [b.to_dict() for b in self.births],
            "merges": self.merges,
            "splits": self.splits,
            "disappearances": self.disappearances,
            "retirements": self.retirements,
            "absorptions": self.absorptions,
            "revivals": self.revivals,
            "document_remap": {str(k): v for k, v in sorted(self.document_remap.items())},
        }


class TopicLedger:
    """Durable, versioned record of every stable topic identifier ever assigned.

    The file is plain JSON and carries a ``fingerprint`` over its content so that
    independent processes can prove they read the same history.

    Parameters
    ----------
    path : str, optional
        Ledger path. ``None`` keeps the ledger in memory only.

    """

    def __init__(self, path=None, payload=None):
        self.path = path
        self._payload = payload if payload is not None else self._empty_payload()

    @staticmethod
    def _empty_payload():
        return {
            "format": LEDGER_FORMAT,
            "version": LEDGER_VERSION,
            "batches": [],
            "next_uid": 1,
        }

    @staticmethod
    def _validate(payload):
        if not isinstance(payload, dict):
            raise LedgerError("ledger root must be a JSON object")
        if payload.get("format") != LEDGER_FORMAT:
            raise LedgerError("not a %r ledger (format=%r)" % (LEDGER_FORMAT, payload.get("format")))
        if payload.get("version") != LEDGER_VERSION:
            raise LedgerError("unsupported ledger version %r (this code understands %r)"
                              % (payload.get("version"), LEDGER_VERSION))
        for key in ("batches", "next_uid"):
            if key not in payload:
                raise LedgerError("ledger is missing required key %r" % (key,))
        if not isinstance(payload["batches"], list):
            raise LedgerError("ledger key 'batches' must be a list")
        return payload

    # ------------------------------------------------------------------ IO
    @classmethod
    def load(cls, path):
        """Read and validate the ledger at ``path``.

        Raises
        ------
        LedgerError
            If the file is absent, is not valid JSON, or has the wrong
            format/version. The message names the path.

        """
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError as err:
            raise LedgerError("ledger %r does not exist" % (path,)) from err
        except (OSError, ValueError) as err:
            raise LedgerError("ledger %r is unreadable or not valid JSON: %s" % (path, err)) from err
        return cls(path=path, payload=cls._validate(payload))

    @classmethod
    def open(cls, path):
        """Load ``path`` if it exists, otherwise start a new in-memory ledger bound to it."""
        if path and os.path.exists(path):
            return cls.load(path)
        return cls(path=path)

    @classmethod
    def read_only(cls, path):
        """Load an independent copy of the ledger for inspection."""
        loaded = cls.load(path)
        return cls(path=None, payload=json.loads(json.dumps(loaded._payload)))

    def save(self):
        """Atomically persist the ledger. Raises :class:`LedgerWriteError` on failure."""
        if not self.path:
            return
        self._payload["fingerprint"] = _fingerprint(
            {k: v for k, v in self._payload.items() if k != "fingerprint"})
        _atomic_write_json(self.path, self._payload)

    # ------------------------------------------------------------------ views
    @property
    def batch_ids(self):
        """Identifiers of all recorded batches, oldest first."""
        return [batch["batch_id"] for batch in self._payload["batches"]]

    @property
    def last_batch_id(self):
        """Identifier of the most recent batch, or ``None``."""
        return self.batch_ids[-1] if self._payload["batches"] else None

    def has_batch(self, batch_id):
        """Whether ``batch_id`` has already been recorded."""
        return batch_id in self.batch_ids

    def next_batch_id(self):
        """Deterministic default identifier for the next batch."""
        return "batch-%03d" % (len(self._payload["batches"]) + 1)

    def current_identities(self):
        """All identities as of the latest batch, active ones first, then by local index."""
        if not self._payload["batches"]:
            return []
        identities = [TopicIdentity.from_dict(record)
                      for record in self._payload["batches"][-1]["identities"].values()]
        identities.sort(key=lambda i: (i.local_index is None,
                                       i.local_index if i.local_index is not None else -1,
                                       i.topic_uid))
        return identities

    def dormant_identities(self):
        """Identities that disappeared earlier but have not yet been retired.

        Their recorded signature is retained so a later batch can revive the same
        identifier by content instead of allocating a new number.
        """
        if not self._payload["batches"]:
            return []
        identities = []
        for record in self._payload["batches"][-1]["identities"].values():
            identity = TopicIdentity.from_dict(record)
            if identity.active:
                continue
            if record.get("retired") or record.get("merged_into") or record.get("absorbed_into"):
                continue
            if identity.signature is not None:
                identities.append(identity)
        identities.sort(key=lambda i: i.topic_uid)
        return identities

    def identities_for_batch(self, batch_id):
        """Return ``{topic_uid: identity_dict}`` as recorded for ``batch_id``.

        Raises
        ------
        LedgerError
            If ``batch_id`` is not present, naming both the requested batch and the
            batches that *are* present.

        """
        for batch in self._payload["batches"]:
            if batch["batch_id"] == batch_id:
                return batch["identities"]
        raise LedgerError("batch %r is not in the ledger (recorded batches: %s)"
                          % (batch_id, ", ".join(self.batch_ids) or "none"))

    # ------------------------------------------------------------------ mutation
    def apply(self, result, config):
        """Append ``result`` as a new batch and return the updated identity records."""
        previous = {identity.topic_uid: identity for identity in self.current_identities()}
        identities = {}

        for match in result.matches:
            uid = match.old.topic_uid
            old = previous.get(uid)
            identities[uid] = TopicIdentity(
                topic_uid=uid,
                label=match.old.label,
                origin_batch=old.origin_batch if old else match.old.origin_batch,
                local_index=match.new_local_index,
                revision=(old.revision + 1 if old else match.old.revision + 1),
                missing=0,
                history=list(old.history if old else match.old.history) + [{
                    "batch_id": result.batch_id,
                    "status": STATUS_MATCHED,
                    "local_index": match.new_local_index,
                    "similarity": round(match.similarity, 6),
                    "drift": None if match.drift is None else round(match.drift, 6),
                    "drift_status": match.drift_status,
                }],
                signature=result.signature_map.get(match.new_local_index),
            ).to_dict()

        for revival in result.revivals:
            uid = revival["topic_uid"]
            old = previous.get(uid)
            identities[uid] = TopicIdentity(
                topic_uid=uid,
                label=old.label if old else revival["label"],
                origin_batch=old.origin_batch if old else result.batch_id,
                local_index=revival["new_local_index"],
                revision=(old.revision if old else 0),
                missing=0,
                history=list(old.history if old else []) + [{
                    "batch_id": result.batch_id,
                    "status": STATUS_MATCHED,
                    "revived": True,
                    "local_index": revival["new_local_index"],
                    "similarity": revival["similarity"],
                }],
                signature=result.signature_map.get(revival["new_local_index"]),
            ).to_dict()

        for merge in result.merges:
            uid = merge["absorbed"]
            old = previous.get(uid)
            identities[uid] = TopicIdentity(
                topic_uid=uid,
                label=old.label if old else uid,
                origin_batch=old.origin_batch if old else result.batch_id,
                local_index=None,
                revision=old.revision if old else 0,
                missing=0,
                history=list(old.history if old else []) + [{
                    "batch_id": result.batch_id,
                    "status": STATUS_MERGED,
                    "into": merge["survivor"],
                    "new_local_index": merge["new_local_index"],
                }],
                signature=None,
            ).to_dict()
            identities[uid]["merged_into"] = merge["survivor"]

        for split in result.splits:
            index = split["secondary_new_local_index"]
            identities[split["secondary"]] = TopicIdentity(
                topic_uid=split["secondary"],
                label=_make_label(result.signature_map[index].words),
                origin_batch=result.batch_id,
                local_index=index,
                revision=0,
                missing=0,
                history=[{"batch_id": result.batch_id, "status": STATUS_BORN,
                          "split_from": split["parent"]}],
                signature=result.signature_map[index],
            ).to_dict()

        for birth in result.births:
            if birth.topic_uid in identities:
                continue
            index = _index_of(result.local_to_uid, birth.topic_uid)
            identities[birth.topic_uid] = TopicIdentity(
                topic_uid=birth.topic_uid,
                label=birth.label,
                origin_batch=result.batch_id,
                local_index=index,
                revision=0,
                missing=0,
                history=[{"batch_id": result.batch_id, "status": STATUS_BORN}],
                signature=result.signature_map.get(index),
            ).to_dict()

        for absorption in result.absorptions:
            uid = absorption["absorbed"]
            old = previous.get(uid)
            identities.setdefault(uid, TopicIdentity(
                topic_uid=uid,
                label=old.label if old else uid,
                origin_batch=old.origin_batch if old else result.batch_id,
                local_index=None,
                revision=old.revision if old else 0,
                missing=0,
                history=list(old.history if old else []) + [{
                    "batch_id": result.batch_id,
                    "status": STATUS_ABSORBED,
                    "into": absorption["into"],
                }],
                signature=None,
            ).to_dict())
            identities[uid]["absorbed_into"] = absorption["into"]

        for entry in result.disappearances + result.retirements:
            uid = entry["topic_uid"]
            if uid in identities:
                continue
            old = previous.get(uid)
            is_retired = bool(entry.get("retired"))
            identities[uid] = TopicIdentity(
                topic_uid=uid,
                label=old.label if old else entry.get("label", uid),
                origin_batch=old.origin_batch if old else result.batch_id,
                local_index=None,
                revision=old.revision if old else 0,
                missing=(old.missing + 1) if old else 1,
                history=list(old.history if old else []) + [{
                    "batch_id": result.batch_id,
                    "status": STATUS_RETIRED if is_retired else STATUS_DISAPPEARED,
                    "destination": entry.get("destination"),
                }],
                # Keep the last known content so the identifier can be revived by
                # content in a later batch instead of silently allocating a new one.
                signature=old.signature if old else None,
            ).to_dict()
            identities[uid]["dormant"] = not is_retired
            identities[uid]["retired"] = is_retired

        self._payload["batches"].append({
            "batch_id": result.batch_id,
            "previous_batch_id": result.previous_batch_id,
            "assigned_at": time.time(),
            "num_topics": len(result.local_to_uid),
            "config": config.to_dict() if config is not None else None,
            "identities": identities,
            "result": result.to_report(),
        })
        self._payload["next_uid"] = max(
            int(self._payload["next_uid"]), _max_uid_number(identities) + 1)
        return identities

    def allocate_identities(self, count, batch_id, labels, reserved=None):
        """Allocate ``count`` brand-new identifiers without committing them."""
        number = int(self._payload["next_uid"])
        used = {
            record["topic_uid"]
            for batch in self._payload["batches"] for record in batch["identities"].values()
        }
        if reserved:
            used |= set(reserved)
        allocated = []
        for label in labels:
            uid = "%s%04d" % (UID_PREFIX, number)
            while uid in used:
                number += 1
                uid = "%s%04d" % (UID_PREFIX, number)
            used.add(uid)
            allocated.append(TopicIdentity(uid, label, origin_batch=batch_id))
            number += 1
        return allocated

    def verify(self):
        """Return a list of human-readable internal inconsistencies (empty if sound)."""
        problems = []
        seen_batches = set()
        batches = self._payload["batches"]
        for index, batch in enumerate(batches):
            batch_id = batch.get("batch_id")
            if batch_id in seen_batches:
                problems.append("duplicate batch_id %r" % (batch_id,))
            seen_batches.add(batch_id)
            if index and batch.get("previous_batch_id") != batches[index - 1]["batch_id"]:
                problems.append(
                    "batch %r declares previous_batch_id=%r but the previous recorded batch is %r"
                    % (batch_id, batch.get("previous_batch_id"), batches[index - 1]["batch_id"]))
            active = [r for r in batch.get("identities", {}).values() if r.get("local_index") is not None]
            recorded = batch.get("num_topics")
            if recorded is None:
                problems.append("batch %r does not record num_topics" % (batch_id,))
            elif len(active) != recorded:
                problems.append(
                    "batch %r records num_topics=%d but has %d identities with a local index"
                    % (batch_id, recorded, len(active)))
        return problems


def _max_uid_number(identities):
    numbers = [0]
    for uid in identities:
        if uid.startswith(UID_PREFIX):
            suffix = uid[len(UID_PREFIX):]
            if suffix.isdigit():
                numbers.append(int(suffix))
    return max(numbers)


def _index_of(mapping, uid):
    for index, candidate in mapping.items():
        if candidate == uid:
            return index
    return None


class TopicAligner:
    """Align a model's topics to stable identifiers and persist the result.

    Parameters
    ----------
    config : AlignmentConfig
    ledger : TopicLedger, optional
        Ledger to read/write. Defaults to ``TopicLedger.open(config.ledger_path)``,
        i.e. the existing ledger is continued when present.

    """

    def __init__(self, config, ledger=None):
        if not isinstance(config, AlignmentConfig):
            raise TypeError("config must be an AlignmentConfig")
        if not config.enabled:
            raise AlignmentError("topic alignment is disabled in this AlignmentConfig")
        if ledger is None:
            if not config.ledger_path:
                raise ValueError("an enabled AlignmentConfig needs ledger_path or an explicit ledger")
            ledger = TopicLedger.open(config.ledger_path)
        self.config = config
        self.ledger = ledger
        self._allocated = set()

    # ------------------------------------------------------------------ signatures
    def signatures_from_model(self, model, batch_id):
        """Extract one :class:`TopicSignature` per topic from a trained model.

        Only public model output is used (``num_topics``, ``id2word``, ``get_topics``),
        so this works for any model exposing that interface. Terms are ordered by
        ``(-probability, term id)`` so ties never depend on initialisation order.

        """
        topics = np.asarray(model.get_topics(), dtype=np.float64)
        if topics.ndim != 2 or topics.shape[0] != int(model.num_topics):
            raise AlignmentError(
                "model.get_topics() returned shape %r but num_topics is %r"
                % (topics.shape, model.num_topics))
        id2word = model.id2word

        def word_of(term_id):
            try:
                return id2word[term_id]
            except (KeyError, IndexError):
                return str(term_id)

        signatures = []
        for local_index in range(int(model.num_topics)):
            row = topics[local_index]
            order = sorted(range(row.shape[0]), key=lambda t: (-row[t], t))[:self.config.topn]
            words = [(word_of(t), float(row[t])) for t in order if row[t] > 0]
            if not words:
                words = [(word_of(order[0]), 1.0)]
            signatures.append(TopicSignature(words, batch_id=batch_id, local_index=local_index))
        return signatures

    # ------------------------------------------------------------------ align
    def align(self, model, batch_id=None):
        """Align ``model``'s topics and return an :class:`AlignmentResult`.

        The ledger is read but not written; call :meth:`commit` (or use
        :meth:`align_and_commit`) to persist the result.

        Raises
        ------
        LedgerError
            If ``require_previous`` is set and no previous batch exists, if ``batch_id``
            is already recorded, or if the ledger is internally inconsistent (the
            message names the batch and the topic-count difference).
        AlignmentError
            If the model does not expose the expected topic API.

        """
        self._check_previous(batch_id)
        previous = [identity for identity in self.ledger.current_identities() if identity.active]
        previous_batch_id = self.ledger.last_batch_id
        batch_id = batch_id or self.ledger.next_batch_id()

        if self.ledger.has_batch(batch_id):
            raise LedgerError(
                "batch %r is already recorded in the ledger (recorded batches: %s); "
                "use a new batch_id or a fresh ledger"
                % (batch_id, ", ".join(self.ledger.batch_ids) or "none"))

        signatures = self.signatures_from_model(model, batch_id)
        result = AlignmentResult(batch_id, previous_batch_id)
        result.signature_map = {sig.local_index: sig for sig in signatures}
        result.previous_local_to_uid = {
            identity.local_index: identity.topic_uid for identity in previous}

        assignment = self._resolve(previous, signatures, result)
        self._emit(assignment, previous, signatures, result)
        self._apply_max_topics(signatures, result)
        self._apply_disappearances(assignment, previous, signatures, result)

        result.local_to_uid = dict(sorted(result.local_to_uid.items()))
        result.diagnostics.update({
            "similarity_threshold": self.config.similarity_threshold,
            "drift_threshold": self.config.drift_threshold,
            "min_shared_terms": self.config.min_shared_terms,
            "topn": self.config.topn,
            "max_topics": self.config.max_topics,
            "previous_batch": previous_batch_id,
            "previous_topics": len(previous),
            "current_topics": len(signatures),
        })
        return result

    def _check_previous(self, batch_id):
        """Validate that the ledger's history is usable for this alignment."""
        batches = self.ledger._payload["batches"]
        if self.config.require_previous and not batches:
            raise LedgerError(
                "cannot align batch %r: the ledger has no previous batch "
                "(expected at least one recorded batch; check ledger %r)"
                % (batch_id, self.ledger.path))
        if not batches:
            return
        problems = self.ledger.verify()
        if problems:
            raise LedgerError("ledger %r is inconsistent: %s"
                              % (self.ledger.path, "; ".join(problems)))
        last = batches[-1]
        broken = [r for r in last.get("identities", {}).values()
                  if r.get("local_index") is not None and not r.get("signature")]
        if broken:
            raise LedgerError(
                "batch %r records %d active topic(s) with no content signature (%s); "
                "the ledger cannot be aligned against"
                % (last.get("batch_id"), len(broken),
                   ", ".join(sorted(r["topic_uid"] for r in broken))))

    def _resolve(self, previous, signatures, result):
        """Decide, for every previous and every current topic, which identifier it gets.

        The resolution proceeds in five deterministic steps:

        1. Collect every (previous, current) pair whose content similarity meets the
           threshold, plus pairs between *dormant* identities (disappeared earlier, grace
           not exhausted) and current topics.
        2. Greedily take active pairs in descending similarity order, each previous topic
           and each current topic at most once. This yields one-to-one *core* matches and
           is independent of training order.
        3. Dormant identities may reclaim an unclaimed current topic (revival); an active
           match always wins a tie.
        4. For a previous topic that also resembles further unclaimed current topics, the
           parent identifier is moved to the branch with the larger share of the old
           topic's mass; every other branch becomes a split secondary (new number).
        5. A previous topic with no core match is either merged into the current topic it
           resembles most (which a stronger claim already owns), or recorded as
           disappeared.

        Returns
        -------
        dict
            ``owners`` (``{new_index: candidate}``), ``merges``, ``secondaries``,
            ``revivals``, ``merged_old``, ``disappeared_old`` and ``by_old``.

        """
        by_old = defaultdict(list)
        candidates = []
        for identity in previous:
            if identity.signature is None:
                continue
            for signature in signatures:
                cosine, n_shared, cov_old, cov_new = identity.signature.shared_info(signature)
                if cosine < self.config.similarity_threshold:
                    continue
                candidate = {
                    "uid": identity.topic_uid,
                    "old_index": identity.local_index,
                    "new_index": signature.local_index,
                    "cosine": cosine,
                    "n_shared": n_shared,
                    "coverage_old": cov_old,
                    "coverage_new": cov_new,
                    "revived": False,
                }
                candidates.append(candidate)
                by_old[identity.local_index].append(candidate)

        # Step 2: one-to-one core matching among active identities.
        ordered = sorted(
            candidates,
            key=lambda c: (-c["cosine"], -c["n_shared"], c["old_index"], c["new_index"]))
        core_old = {}
        core_new = {}
        for candidate in ordered:
            if candidate["old_index"] in core_old or candidate["new_index"] in core_new:
                continue
            core_old[candidate["old_index"]] = candidate
            core_new[candidate["new_index"]] = candidate

        # Step 3: dormant identities may reclaim topics nobody active claimed.
        dormant_claims = []
        for identity in self.ledger.dormant_identities():
            if identity.signature is None:
                continue
            for signature in signatures:
                cosine, n_shared, cov_old, cov_new = identity.signature.shared_info(signature)
                if cosine < self.config.similarity_threshold:
                    continue
                dormant_claims.append({
                    "uid": identity.topic_uid,
                    "old_index": None,
                    "new_index": signature.local_index,
                    "cosine": cosine,
                    "n_shared": n_shared,
                    "coverage_old": cov_old,
                    "coverage_new": cov_new,
                    "revived": True,
                })
        dormant_claims.sort(key=lambda c: (-c["cosine"], -c["n_shared"], c["uid"], c["new_index"]))
        revivals = []
        for claim in dormant_claims:
            if claim["new_index"] in core_new:
                continue
            if any(claim["uid"] == other["uid"] for other in revivals):
                continue
            core_new[claim["new_index"]] = claim
            revivals.append(claim)

        # Step 4: split branches and parent-follows-larger-share.
        secondaries = []
        claimed_branches = set(core_new)
        for old_index in sorted(core_old):
            core = core_old[old_index]
            unclaimed = [
                c for c in by_old[old_index]
                if c["new_index"] not in claimed_branches and c["new_index"] != core["new_index"]
            ]
            if not unclaimed:
                continue
            branches = [core] + unclaimed
            parent = max(branches, key=lambda c: (c["coverage_old"], c["cosine"], -c["new_index"]))
            for branch in branches:
                if branch is parent:
                    continue
                secondaries.append(branch)
                claimed_branches.add(branch["new_index"])
            if parent is not core:
                # Move the parent identifier to the larger branch; the old core target
                # stays claimed as the secondary branch recorded above.
                del core_new[core["new_index"]]
                core_new[parent["new_index"]] = parent
                core_old[old_index] = parent
                claimed_branches.add(parent["new_index"])

        # Step 5: merges and disappearances.
        merges = []
        merged_old = set()
        disappeared_old = []
        for identity in previous:
            old_index = identity.local_index
            if old_index in core_old:
                continue
            claimed = [c for c in by_old.get(old_index, []) if c["new_index"] in core_new]
            if claimed:
                # Every similar current topic is already owned by a stronger claim.
                loser = max(claimed, key=lambda c: (c["cosine"], c["coverage_old"], -c["new_index"]))
                merges.append(loser)
                merged_old.add(old_index)
            else:
                disappeared_old.append(old_index)

        result.diagnostics["candidate_pairs"] = len(candidates)
        result.diagnostics["conflicts"] = {
            "merge_candidates": len(merges),
            "split_candidates": len(secondaries),
        }
        result.diagnostics["revivals"] = len(revivals)
        return {
            "owners": dict(core_new),
            "merges": merges,
            "secondaries": secondaries,
            "revivals": revivals,
            "merged_old": merged_old,
            "disappeared_old": sorted(disappeared_old),
            "by_old": by_old,
        }

    def _emit(self, assignment, previous, signatures, result):
        """Turn the resolution into matches, revivals, merges, splits, births and remaps."""
        by_local = {identity.local_index: identity for identity in previous}
        dormant_by_uid = {identity.topic_uid: identity for identity in self.ledger.dormant_identities()}

        for new_index, owner in sorted(assignment["owners"].items()):
            if owner["revived"]:
                identity = dormant_by_uid[owner["uid"]]
                result.revivals.append({
                    "topic_uid": identity.topic_uid,
                    "label": identity.label,
                    "new_local_index": new_index,
                    "similarity": round(owner["cosine"], 6),
                    "reason": "a dormant identifier reclaimed an unclaimed current topic by content",
                })
                result.local_to_uid[new_index] = identity.topic_uid
                continue
            identity = by_local[owner["old_index"]]
            drift, status = drift_score(
                identity.signature, signatures[new_index],
                min_shared_terms=self.config.min_shared_terms)
            if drift is not None and drift > self.config.drift_threshold:
                status = DRIFT_THRESHOLD_EXCEEDED
            result.matches.append(TopicMatch(
                old=identity,
                new_local_index=new_index,
                similarity=owner["cosine"],
                n_shared_terms=owner["n_shared"],
                drift=drift,
                drift_status=status,
                evidence={
                    "coverage_previous": round(owner["coverage_old"], 6),
                    "coverage_current": round(owner["coverage_new"], 6),
                    "n_shared_terms": owner["n_shared"],
                },
            ))
            result.local_to_uid[new_index] = identity.topic_uid
            result.document_remap[identity.local_index] = new_index

        for loser in sorted(assignment["merges"],
                            key=lambda c: (c["new_index"], -c["cosine"], c["old_index"])):
            identity = by_local[loser["old_index"]]
            survivor_uid = result.local_to_uid[loser["new_index"]]
            result.merges.append({
                "survivor": survivor_uid,
                "absorbed": identity.topic_uid,
                "new_local_index": loser["new_index"],
                "similarity": round(loser["cosine"], 6),
                "reason": "competing claim on one current topic; the weaker claim is merged "
                          "and its document mass folded into the survivor",
            })
            result.document_remap[identity.local_index] = loser["new_index"]

        labels = [_make_label(signatures[c["new_index"]].words)
                  for c in assignment["secondaries"]]
        secondaries = self.ledger.allocate_identities(
            len(labels), result.batch_id, labels, reserved=self._allocated)
        for candidate, identity in zip(assignment["secondaries"], secondaries):
            self._allocated.add(identity.topic_uid)
            new_index = candidate["new_index"]
            parent = by_local[candidate["old_index"]]
            parent_primary = result.document_remap[parent.local_index]
            parent_share = next(
                c["coverage_old"] for c in assignment["by_old"][parent.local_index]
                if c["new_index"] == parent_primary)
            result.births.append(identity)
            result.local_to_uid[new_index] = identity.topic_uid
            result.splits.append({
                "parent": parent.topic_uid,
                "primary_new_local_index": parent_primary,
                "secondary": identity.topic_uid,
                "secondary_new_local_index": new_index,
                "parent_share": round(parent_share, 6),
                "secondary_share": round(candidate["coverage_old"], 6),
                "reason": "one previous topic's vocabulary spread across several current "
                          "topics; the branch with the larger share keeps the parent identifier",
            })

        unassigned = [sig.local_index for sig in signatures
                      if sig.local_index not in result.local_to_uid]
        labels = [_make_label(signatures[index].words) for index in unassigned]
        births = self.ledger.allocate_identities(
            len(labels), result.batch_id, labels, reserved=self._allocated)
        for index, identity in zip(unassigned, births):
            self._allocated.add(identity.topic_uid)
            result.births.append(identity)
            result.local_to_uid[index] = identity.topic_uid

    def _apply_max_topics(self, signatures, result):
        """Absorb the most redundant topics until ``max_topics`` is respected."""
        cap = self.config.max_topics
        if cap is None:
            return
        while len(result.local_to_uid) > cap:
            indexes = sorted(result.local_to_uid)
            ranked = []
            for index in indexes:
                best_other, best_similarity = None, -1.0
                for other in indexes:
                    if other == index:
                        continue
                    similarity = signatures[index].cosine(signatures[other])
                    if similarity > best_similarity:
                        best_other, best_similarity = other, similarity
                ranked.append((-best_similarity, -index, index, best_other))
            ranked.sort()
            _, _, index, target = ranked[0]
            uid = result.local_to_uid.pop(index)
            absorbed_match = next((m for m in result.matches if m.old.topic_uid == uid), None)
            if absorbed_match is not None:
                result.matches.remove(absorbed_match)
                result.document_remap[absorbed_match.old.local_index] = target
                previous_local = absorbed_match.old.local_index
            else:
                previous_local = None
            result.births = [birth for birth in result.births if birth.topic_uid != uid]
            secondary_split = next((s for s in result.splits if s["secondary"] == uid), None)
            if secondary_split is not None:
                result.splits.remove(secondary_split)
            result.absorptions.append({
                "absorbed": uid,
                "absorbed_new_local_index": index,
                "previous_local_index": previous_local,
                "into": result.local_to_uid[target],
                "into_new_local_index": target,
                "similarity": round(max(-ranked[0][0], 0.0), 6),
                "reason": "max_topics=%d exceeded; most redundant topic absorbed by "
                          "content similarity" % cap,
            })

    def _apply_disappearances(self, assignment, previous, signatures, result):
        """Record an explicit destination for every previous topic without a successor."""
        by_local = {identity.local_index: identity for identity in previous}
        for old_index in assignment["disappeared_old"]:
            identity = by_local[old_index]
            entry = {
                "topic_uid": identity.topic_uid,
                "label": identity.label,
                "previous_local_index": old_index,
                "destination": None,
                "reason": "no current topic met the similarity threshold (%.3f)"
                          % self.config.similarity_threshold,
                "retired": False,
            }
            destination = self._closest_index(identity.signature, signatures, result.local_to_uid)
            if destination is not None:
                entry["destination"] = destination
                entry["reason"] += "; document mass folded into the most similar surviving topic"
                result.document_remap[old_index] = destination
            else:
                result.document_remap[old_index] = None
                entry["reason"] += "; no surviving topic exists, its document mass has no destination"
            if identity.missing + 1 > self.config.retired_grace:
                entry["retired"] = True
                result.retirements.append(entry)
            else:
                result.disappearances.append(entry)

    @staticmethod
    def _closest_index(signature, signatures, local_to_uid):
        if signature is None or not local_to_uid:
            return None
        best_index, best_score = None, -1.0
        for index in sorted(local_to_uid):
            score = signature.cosine(signatures[index])
            if score > best_score:
                best_index, best_score = index, score
        return best_index

    # ------------------------------------------------------------------ commit
    def commit(self, result):
        """Persist ``result``. Raises :class:`LedgerWriteError` if the write fails."""
        if not isinstance(result, AlignmentResult):
            raise TypeError("result must be an AlignmentResult")
        if self.ledger.has_batch(result.batch_id):
            raise LedgerError("batch %r is already recorded" % (result.batch_id,))
        self.ledger.apply(result, self.config)
        try:
            self.ledger.save()
        except LedgerWriteError:
            self.ledger._payload["batches"].pop()
            raise

    def align_and_commit(self, model, batch_id=None):
        """Align then persist; returns the :class:`AlignmentResult`."""
        result = self.align(model, batch_id=batch_id)
        self.commit(result)
        return result


def align_topics(model, config=None, batch_id=None, ledger=None):
    """Convenience wrapper. Returns ``None`` when ``config`` is ``None`` or disabled.

    With alignment off it is a no-op, so existing training and topic output are
    untouched.
    """
    if config is None or not config.enabled:
        return None
    return TopicAligner(config, ledger=ledger).align_and_commit(model, batch_id=batch_id)
