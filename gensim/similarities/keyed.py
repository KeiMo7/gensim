#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Identity-addressed deletion, replacement and compaction for the sharded similarity index.

The stock :class:`~gensim.similarities.docsim.Similarity` index is append-only: a document's
position *is* its identity, so the only way to remove a document is to destroy and rebuild the
whole index.  This module adds an opt-in keyed layer on top, via
:class:`~gensim.similarities.keyed.KeyedSimilarity`:

* documents are registered under a **caller-supplied identity** (`doc_id`), validated unique;
* :meth:`~gensim.similarities.keyed.KeyedSimilarity.delete_documents` marks tombstones
  (no rewrite on the query path) and the space is reclaimed asynchronously by compaction;
* :meth:`~gensim.similarities.keyed.KeyedSimilarity.replace_document` swaps the content of an
  existing identity in place: the position and the external identity stay the same;
* deleted identities are not re-usable for a configurable number of mutation rounds, so stale
  external references cannot silently point at a different document;
* once the tombstone ratio crosses `tombstone_ratio`, the index is compacted in the background.
  Compaction is journal-based: after a crash the index either rolls back to the pre-compaction
  state or rolls the compaction forward -- never anything in between;
* all mutation state lives in a single sidecar file (`<output_prefix>.keyed`), written
  atomically on every mutation, so deletes/replaces survive a process crash;
* queries never mutate shared state: concurrent readers observe a consistent snapshot while a
  single writer deletes/replaces;
* with ``keyed=False`` (the default) the class behaves exactly like
  :class:`~gensim.similarities.docsim.Similarity`.

Logging in this module never emits document content or user identifiers -- only document
identities and counters.

"""
import json
import logging
import os
import pickle as _stdlib_pickle
import threading

import numpy
import scipy.sparse

from gensim import matutils, utils
from gensim.similarities.docsim import Similarity, query_shard

logger = logging.getLogger(__name__)

#: Name of the sidecar file that stores the keyed state, relative to `output_prefix`.
SIDECAR_SUFFIX = '.keyed'

#: Name of the compaction journal, relative to `output_prefix`.
JOURNAL_SUFFIX = '.compact.journal'

#: Environment variable used by tests to simulate a crash at a given compaction stage.
CRASH_AT_ENVVAR = 'GENSIM_KEYED_CRASH_AT'

#: Environment variable used by tests to simulate a storage failure during reclaim.
FAIL_RECLAIM_ENVVAR = 'GENSIM_KEYED_FAIL_RECLAIM'

#: Attributes that must never be pickled (runtime-only, recreated on load).
_RUNTIME_ATTRS = ('_lock', '_write_lock', '_compact_thread')


class KeyedState(object):
    """Immutable snapshot of the keyed state. Mutations replace, never modify, these objects,
    so a reader that grabbed a snapshot sees a consistent view for the whole query."""

    __slots__ = ('id2pos', 'pos2id', 'tombstones', 'replacements', 'shards', 'ntotal')

    def __init__(self, id2pos, pos2id, tombstones, replacements, shards, ntotal):
        self.id2pos = id2pos
        self.pos2id = pos2id
        self.tombstones = tombstones
        self.replacements = replacements
        self.shards = shards
        self.ntotal = ntotal


class KeyedSimilarity(Similarity):
    """A :class:`~gensim.similarities.docsim.Similarity` index with optional identity-based
    deletion and replacement.

    Parameters
    ----------
    output_prefix : str
        Prefix for shard filename. If None, a random filename in temp will be used.
    corpus : iterable of list of (int, number), optional
        Corpus in streamed Gensim bag-of-words format.
    num_features : int
        Size of the dictionary (number of features).
    num_best : int, optional
        If set, return only the `num_best` most similar documents. In keyed mode the result
        contains ``(doc_id, similarity)`` pairs and always contains `num_best` live documents
        (as long as enough live documents with non-zero similarity exist) -- deletions do not
        eat into the result count.
    chunksize : int, optional
        Size of query chunks.
    shardsize : int, optional
        Maximum shard size, in documents.
    norm : {'l1', 'l2'}, optional
        Normalization to use.
    keyed : bool, optional
        Enable the keyed layer (identity-based delete/replace/compact). Default False: the
        index behaves exactly like :class:`~gensim.similarities.docsim.Similarity`.
    doc_ids : list of str, optional
        Caller-supplied identities, one per document in `corpus`. Required iff `keyed` is True
        and `corpus` is given.
    tombstone_ratio : float, optional
        Trigger asynchronous compaction once deleted positions reach this fraction of the index.
    id_reuse_rounds : int, optional
        A deleted identity may not be re-registered within this many mutation rounds.
    auto_compact : bool, optional
        Compact automatically (in a background thread) once `tombstone_ratio` is reached.

    Notes
    -----
    With ``keyed=True``:

    * query results never contain deleted identities, on all three query paths (single
      document, batch corpus, self-query via iteration);
    * with ``num_best=None`` the result arrays contain one column per *live* document, in
      ascending position order; use :meth:`~gensim.similarities.keyed.KeyedSimilarity.live_doc_ids`
      for the identities aligned with the columns of the last query;
    * :func:`len` still reports the physical number of positions (including tombstones);
      use :attr:`~gensim.similarities.keyed.KeyedSimilarity.live_count` for the live count.

    """

    def __init__(self, output_prefix, corpus, num_features, num_best=None, chunksize=256,
                 shardsize=32768, norm='l2', keyed=False, doc_ids=None, tombstone_ratio=0.2,
                 id_reuse_rounds=2, auto_compact=True):
        self.tombstone_ratio = tombstone_ratio
        self.id_reuse_rounds = id_reuse_rounds
        self.auto_compact = auto_compact
        self._init_keyed_runtime()
        self._init_keyed_state()
        super(KeyedSimilarity, self).__init__(
            output_prefix, None, num_features, num_best=num_best, chunksize=chunksize,
            shardsize=shardsize, norm=norm,
        )
        self._recover_compaction()
        self._load_sidecar()
        if keyed and not self.keyed:
            self.enable_keyed()
        if corpus is not None:
            self.add_documents(corpus, doc_ids=doc_ids)

    #
    # state management
    #

    def _init_keyed_runtime(self):
        """(Re)create runtime-only objects. Called from __init__ and after unpickling."""
        self._lock = threading.RLock()
        self._write_lock = threading.RLock()
        self._compact_thread = None
        self._compact_crash_at = os.environ.get(CRASH_AT_ENVVAR) or None
        self._fail_reclaim = os.environ.get(FAIL_RECLAIM_ENVVAR) or None

    def _init_keyed_state(self):
        """Reset the keyed state to 'disabled / empty'."""
        self.keyed = False
        self.id2pos = {}
        self.pos2id = []
        self.tombstones = set()
        self.replacements = {}
        self.retired = {}
        self.round = 0
        self._state_version = 0
        self.stats = {
            'added': 0,
            'deleted': 0,
            'replaced': 0,
            'delete_misses': 0,
            'compactions': 0,
            'reclaim_failures': 0,
        }
        self.last_reclaim_error = None

    def __getstate__(self):
        result = self.__dict__.copy()
        for attr in _RUNTIME_ATTRS:
            result.pop(attr, None)
        return result

    def _snapshot(self):
        """Grab a consistent, immutable view of the keyed state (copy-on-write)."""
        with self._lock:
            shards = list(self.shards)
            return KeyedState(
                id2pos=self.id2pos,
                pos2id=self.pos2id,
                tombstones=self.tombstones,
                replacements=self.replacements,
                shards=shards,
                ntotal=sum(len(shard) for shard in shards),
            )

    def _require_keyed(self):
        if not self.keyed:
            raise RuntimeError("keyed mode is disabled: construct with keyed=True or call enable_keyed()")

    def enable_keyed(self, doc_ids=None):
        """Enable the keyed layer on this index.

        Parameters
        ----------
        doc_ids : list of str, optional
            Identities for the documents already in the index. Required when the index is not
            empty: identities are caller-supplied, never invented. Must be unique and exactly
            one per existing document.

        """
        with self._lock:
            if self.keyed:
                raise RuntimeError("keyed mode is already enabled")
            ndocs = len(self)
            if doc_ids is None:
                if ndocs:
                    raise ValueError(
                        "cannot enable keyed mode on a non-empty index without caller-supplied "
                        "doc_ids (%d documents)" % ndocs
                    )
                doc_ids = []
            doc_ids = list(doc_ids)
            if len(doc_ids) != ndocs:
                raise ValueError(
                    "doc_ids length mismatch: got %d identities for %d indexed documents"
                    % (len(doc_ids), ndocs)
                )
            duplicates = sorted({d for d in doc_ids if doc_ids.count(d) > 1})
            if duplicates:
                raise ValueError("duplicate identities in doc_ids: %r" % (duplicates,))
            self.id2pos = dict(zip(doc_ids, range(ndocs)))
            self.pos2id = list(doc_ids)
            self.keyed = True
            self._state_version += 1
            self._persist_locked()
        logger.info("keyed mode enabled with %d identities", ndocs)

    @classmethod
    def promote(cls, index, doc_ids):
        """Promote a plain :class:`~gensim.similarities.docsim.Similarity` index (e.g. one
        loaded from disk, in the old format) to a keyed index.

        Parameters
        ----------
        index : :class:`~gensim.similarities.docsim.Similarity`
            The existing index. Its shards, mmap and lazy-loading behaviour are kept as-is.
        doc_ids : list of str
            Caller-supplied identities, one per document already in `index`.

        Returns
        -------
        :class:`~gensim.similarities.keyed.KeyedSimilarity`
            The same index, with the keyed layer enabled.

        """
        new = cls.__new__(cls)
        new.__dict__.update(index.__dict__)
        new.tombstone_ratio = getattr(new, 'tombstone_ratio', 0.2)
        new.id_reuse_rounds = getattr(new, 'id_reuse_rounds', 2)
        new.auto_compact = getattr(new, 'auto_compact', True)
        new._init_keyed_runtime()
        new._init_keyed_state()
        new._recover_compaction()
        new._load_sidecar()
        if not new.keyed:
            new.enable_keyed(doc_ids)
        return new

    #
    # persistence: sidecar + compaction journal
    #

    def _sidecar_fname(self):
        return self.output_prefix + SIDECAR_SUFFIX

    def _journal_fname(self):
        return self.output_prefix + JOURNAL_SUFFIX

    def _shards_meta(self):
        """Serializable description of the current shard layout."""
        return [
            {
                'fullname': shard.fullname(),
                'length': len(shard),
                'cls': shard.cls.__name__,
                'num_nnz': getattr(shard, 'num_nnz', None),
            }
            for shard in self.shards
        ]

    def _shards_from_meta(self, meta):
        """Rebuild shard proxies from their serialized description (lazy mmap on first use)."""
        from gensim.similarities.docsim import MatrixSimilarity, Shard, SparseMatrixSimilarity
        classes = {
            'MatrixSimilarity': MatrixSimilarity,
            'SparseMatrixSimilarity': SparseMatrixSimilarity,
        }
        shards = []
        for entry in meta:
            shard = Shard.__new__(Shard)
            shard.dirname, shard.fname = os.path.split(entry['fullname'])
            shard.length = entry['length']
            shard.cls = classes[entry['cls']]
            shard.num_best = self.num_best
            shard.num_nnz = entry.get('num_nnz')
            shards.append(shard)
        return shards

    def _persist_locked(self):
        """Write the keyed state to disk atomically. Caller must hold ``self._lock``.

        The write is tmp-file + rename, so a crash either leaves the old state or the new
        state, never a torn file. The sidecar is the authoritative description of both the
        keyed state and the shard layout: after a crash without save(), the index is
        reconstructed from it. Only identities and counters are written -- never content
        metadata beyond the replacement vectors themselves.
        """
        if not self.keyed:
            return
        state = {
            'format': 1,
            'version': self._state_version,
            'id2pos': self.id2pos,
            'pos2id': self.pos2id,
            'tombstones': self.tombstones,
            'replacements': self.replacements,
            'retired': self.retired,
            'round': self.round,
            'stats': self.stats,
            'shards_meta': self._shards_meta(),
        }
        fname = self._sidecar_fname()
        tmp = fname + '.tmp'
        with open(tmp, 'wb') as fout:
            _stdlib_pickle.dump(state, fout, protocol=utils.PICKLE_PROTOCOL)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp, fname)

    def _load_sidecar(self):
        """Load keyed state from the sidecar file, if it is newer than the pickled state."""
        fname = self._sidecar_fname()
        if not os.path.exists(fname):
            return
        with open(fname, 'rb') as fin:
            state = _stdlib_pickle.load(fin)
        if state.get('version', 0) <= getattr(self, '_state_version', 0):
            return
        with self._lock:
            self.keyed = True
            self.id2pos = state['id2pos']
            self.pos2id = state['pos2id']
            self.tombstones = state['tombstones']
            self.replacements = state['replacements']
            self.retired = state['retired']
            self.round = state['round']
            self.stats.update(state['stats'])
            self._state_version = state['version']
            self.shards = self._shards_from_meta(state.get('shards_meta', []))
            self.fresh_docs, self.fresh_nnz = [], 0
            # documents buffered in memory when the process died are gone, same as with the
            # base index; drop their identities so the state stays self-consistent
            ntotal = sum(len(shard) for shard in self.shards)
            if len(self.pos2id) != ntotal:
                dropped = len(self.pos2id) - ntotal
                if dropped > 0:
                    logger.warning(
                        "discarding %d identities whose documents were not flushed to disk "
                        "before the crash", dropped,
                    )
                self.pos2id = self.pos2id[:ntotal]
                self.id2pos = {doc_id: self.id2pos[doc_id] for doc_id in self.pos2id
                               if doc_id in self.id2pos}
                self.tombstones = {pos for pos in self.tombstones if pos < ntotal}
                self.replacements = {pos: v for pos, v in self.replacements.items() if pos < ntotal}
        logger.warning(
            "recovered keyed state from %s: %d live identities, %d tombstones, %d replacements, "
            "%d shards, round %d",
            fname, len(self.id2pos), len(self.tombstones), len(self.replacements),
            len(self.shards), self.round,
        )

    def save(self, fname=None, *args, **kwargs):
        """Save the index. The keyed state is stored both in the pickle and, authoritatively,
        in the ``<output_prefix>.keyed`` sidecar (which is also written on every mutation, so
        deletes survive a crash even without an explicit save)."""
        with self._lock:
            self._persist_locked()
        super(KeyedSimilarity, self).save(fname, *args, **kwargs)

    @classmethod
    def load(cls, fname, mmap=None):
        """Load an index saved with :meth:`save`. Old indexes saved before the keyed layer
        existed load unchanged (keyed mode stays disabled); enable it afterwards with
        :meth:`enable_keyed` or :meth:`promote`. If a compaction was interrupted by a crash,
        it is rolled back or completed first."""
        obj = super(KeyedSimilarity, cls).load(fname, mmap=mmap)
        if isinstance(obj, KeyedSimilarity):
            obj._init_keyed_runtime()
            obj._recover_compaction()
            obj._load_sidecar()
        return obj

    #
    # mutations
    #

    def _check_new_ids_locked(self, doc_ids):
        """Validate caller-supplied identities for registration. Caller must hold the lock."""
        seen = set()
        duplicates = []
        for doc_id in doc_ids:
            if doc_id in seen:
                duplicates.append(doc_id)
            seen.add(doc_id)
        if duplicates:
            raise ValueError("duplicate identities within the batch: %r" % (sorted(duplicates),))
        conflicts = [d for d in doc_ids if d in self.id2pos]
        if conflicts:
            raise ValueError("identities already registered: %r" % (sorted(conflicts),))
        too_fresh = [
            d for d in doc_ids
            if d in self.retired and self.round - self.retired[d] <= self.id_reuse_rounds
        ]
        if too_fresh:
            raise ValueError(
                "identities deleted too recently to re-register (wait %d more rounds): %r"
                % (self.id_reuse_rounds, sorted(too_fresh))
            )

    def add_documents(self, corpus, doc_ids=None):
        """Extend the index with new documents.

        Parameters
        ----------
        corpus : iterable of list of (int, number)
            Corpus in BoW format.
        doc_ids : list of str, optional
            Caller-supplied identities, one per document. Required in keyed mode; rejected
            otherwise.

        """
        if not self.keyed:
            if doc_ids is not None:
                raise ValueError("doc_ids were supplied but keyed mode is disabled")
            return super(KeyedSimilarity, self).add_documents(corpus)
        if doc_ids is None:
            raise ValueError("keyed mode requires caller-supplied doc_ids for add_documents")
        corpus = list(corpus)
        doc_ids = list(doc_ids)
        if len(corpus) != len(doc_ids):
            raise ValueError(
                "doc_ids length mismatch: got %d identities for %d documents"
                % (len(doc_ids), len(corpus))
            )
        with self._write_lock:
            with self._lock:
                self._check_new_ids_locked(doc_ids)
                base_pos = len(self.fresh_docs) + sum(len(shard) for shard in self.shards)
            super(KeyedSimilarity, self).add_documents(corpus)
            with self._lock:
                new_id2pos = dict(self.id2pos)
                new_pos2id = list(self.pos2id)
                for offset, doc_id in enumerate(doc_ids):
                    new_id2pos[doc_id] = base_pos + offset
                    new_pos2id.append(doc_id)
                self.id2pos = new_id2pos
                self.pos2id = new_pos2id
                self.round += 1
                self._state_version += 1
                self.stats['added'] += len(doc_ids)
                self._persist_locked()
        logger.info("added %d documents, %d live identities", len(doc_ids), len(self.id2pos))

    def delete_document(self, doc_id):
        """Delete a single document by identity. See :meth:`delete_documents`."""
        self.delete_documents([doc_id])

    def delete_documents(self, doc_ids):
        """Delete a batch of documents by identity.

        The batch is atomic: either all identities are marked deleted or none are. Queries
        already in flight keep their consistent snapshot; new queries stop seeing the deleted
        identities immediately after this call returns. The storage is reclaimed later,
        asynchronously, by compaction.

        Parameters
        ----------
        doc_ids : iterable of str
            Identities to delete.

        Raises
        ------
        KeyError
            If any identity is not registered (explicit miss -- never a silent success).
            Nothing is deleted in that case.

        """
        self._require_keyed()
        doc_ids = list(doc_ids)
        with self._lock:
            missing = [d for d in doc_ids if d not in self.id2pos]
            if missing:
                self.stats['delete_misses'] += len(missing)
                self._state_version += 1
                self._persist_locked()
                raise KeyError("delete missed %d unknown identities: %r" % (len(missing), missing))
            positions = {self.id2pos[d] for d in doc_ids}
            doomed = set(doc_ids)
            new_retired = dict(self.retired)
            for doc_id in doomed:
                new_retired[doc_id] = self.round
            self.tombstones = set(self.tombstones) | positions
            self.id2pos = {k: v for k, v in self.id2pos.items() if k not in doomed}
            self.retired = new_retired
            self.replacements = {p: v for p, v in self.replacements.items() if p not in positions}
            self.round += 1
            self._state_version += 1
            self.stats['deleted'] += len(doomed)
            self._persist_locked()
        logger.info(
            "deleted %d documents, %d live identities, %d tombstones",
            len(doomed), len(self.id2pos), len(self.tombstones),
        )
        self._maybe_compact()

    def _convert_doc(self, doc):
        """Convert a document to its stored (normalized, dense) vector form, mirroring
        :meth:`~gensim.similarities.docsim.Similarity.add_documents`. Validates dimensions."""
        if isinstance(doc, numpy.ndarray):
            if doc.ndim != 1 or doc.shape[0] != self.num_features:
                raise ValueError(
                    "dimension mismatch: expected a vector of %d features, got shape %r"
                    % (self.num_features, getattr(doc, 'shape', None))
                )
            return numpy.asarray(doc, dtype=numpy.float32)
        if scipy.sparse.issparse(doc):
            if doc.shape[1] != self.num_features:
                raise ValueError(
                    "dimension mismatch: expected %d features, got %d"
                    % (self.num_features, doc.shape[1])
                )
            return numpy.asarray(doc.toarray(), dtype=numpy.float32).flatten()
        # standard gensim BoW format
        if len(doc):
            max_id = max(termid for termid, _ in doc)
            if max_id >= self.num_features:
                raise ValueError(
                    "dimension mismatch: feature id %d exceeds num_features=%d"
                    % (max_id, self.num_features)
                )
        return matutils.unitvec(matutils.sparse2full(doc, self.num_features), self.norm).astype(numpy.float32)

    def replace_document(self, doc_id, doc):
        """Replace the content of an existing identity. The position and the external identity
        stay the same; queries issued after this call returns see the new content.

        Parameters
        ----------
        doc_id : str
            Identity of the document to replace.
        doc : list of (int, number) or :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
            The new document content.

        Raises
        ------
        KeyError
            If `doc_id` is not registered (explicit miss).
        ValueError
            If the new content's dimensionality conflicts with the index.

        """
        self._require_keyed()
        vector = self._convert_doc(doc)
        with self._lock:
            if doc_id not in self.id2pos:
                raise KeyError("cannot replace unknown identity: %r" % (doc_id,))
            pos = self.id2pos[doc_id]
            new_replacements = dict(self.replacements)
            new_replacements[pos] = vector
            self.replacements = new_replacements
            self.round += 1
            self._state_version += 1
            self.stats['replaced'] += 1
            self._persist_locked()
        logger.info("replaced 1 document at position %d, %d live identities", pos, len(self.id2pos))
        self._maybe_compact()

    #
    # queries
    #

    def _query_vectors(self, query):
        """Convert a query (single doc or corpus) to a list of dense query vectors, applying
        the same normalization rules as the shard query path."""
        is_corpus, query = utils.is_corpus(query)
        is_corpus = is_corpus or hasattr(query, 'ndim') and query.ndim > 1 and query.shape[0] > 1
        docs = query if is_corpus else [query]
        vectors = []
        for doc in docs:
            if isinstance(doc, numpy.ndarray):
                vectors.append(numpy.asarray(doc, dtype=numpy.float32))
            elif scipy.sparse.issparse(doc):
                vectors.append(numpy.asarray(doc.toarray(), dtype=numpy.float32).flatten())
            else:
                if self.norm:
                    doc = matutils.unitvec(doc, self.norm)
                vectors.append(matutils.sparse2full(doc, self.num_features).astype(numpy.float32))
        return vectors, is_corpus

    def __getitem__(self, query):
        """Get similarities of the document (or corpus) `query` to all live documents.

        In keyed mode, deleted identities never appear in the result. With `num_best` set,
        the result is a list of ``(doc_id, similarity)`` pairs with up to `num_best` entries;
        deletions do not eat into the count. With ``num_best=None``, the result contains one
        column per live document (see :meth:`live_doc_ids`).

        With keyed mode disabled, identical to
        :meth:`~gensim.similarities.docsim.Similarity.__getitem__`.
        """
        if not self.keyed:
            return super(KeyedSimilarity, self).__getitem__(query)

        with self._write_lock:
            self.close_shard()  # no-op if no documents added to index since last query
        snap = self._snapshot()
        if snap.ntotal == 0:
            empty = numpy.array([], dtype=numpy.float32)
            if self.num_best is None:
                return empty
            return []

        # The query path is read-only w.r.t. shared state: every shard is asked for the full
        # similarity vector (num_best=None), and clipping/filtering happens per-query below.
        # Concurrent readers therefore never interfere, and each reader works on its own
        # consistent snapshot of the keyed state.
        for shard in snap.shards:
            shard.num_best = None
            shard.normalize = self.norm
            shard.get_index()  # prefetch the mmap now, so a compaction swapping files
            # underneath us can never break this in-flight query
        results = map(query_shard, zip([query] * len(snap.shards), snap.shards))
        full = numpy.hstack(list(results))

        qvecs, _ = self._query_vectors(query)
        repl_sims = [
            {pos: float(numpy.dot(vec, qvec)) for pos, vec in snap.replacements.items()}
            for qvec in qvecs
        ]

        if self.num_best is None:
            return self._apply_full(full, snap, repl_sims)
        return self._apply_topn(full, snap, repl_sims)

    def _apply_full(self, full, snap, repl_sims):
        """Filter a full similarity result down to live positions, replacing stale columns
        of replaced documents with their exact similarities."""
        live = [pos for pos in range(snap.ntotal) if pos not in snap.tombstones]
        out = full[..., live]
        if snap.replacements:
            live_pos = {pos: col for col, pos in enumerate(live)}
            rows = repl_sims if out.ndim > 1 else repl_sims[:1]
            for row_no, sims in enumerate(rows):
                for pos, sim in sims.items():
                    col = live_pos.get(pos)
                    if col is not None:
                        if out.ndim > 1:
                            out[row_no, col] = sim
                        else:
                            out[col] = sim
        self._last_live_positions = live
        return out

    def _apply_topn(self, full, snap, repl_sims):
        """Clip a full similarity result to the `num_best` live documents, as
        ``(doc_id, similarity)`` pairs. Tombstoned positions are excluded, replaced positions
        use their exact (new-content) similarities."""
        rows = full if full.ndim > 1 else [full]
        tombstones = numpy.fromiter(snap.tombstones, dtype=numpy.intp) if snap.tombstones else None
        out = []
        for row_no, row in enumerate(rows):
            sims = numpy.array(row, dtype=numpy.float64)  # copy: never touch the mmap
            if tombstones is not None:
                sims[tombstones] = 0.0
            for pos, sim in repl_sims[row_no].items():
                sims[pos] = sim
            order = numpy.argsort(-numpy.abs(sims), kind='stable')
            top = [pos for pos in order[:self.num_best + len(snap.tombstones) + len(snap.replacements)]
                   if sims[pos] != 0.0][:self.num_best]
            out.append([(snap.pos2id[pos], float(sims[pos])) for pos in top])
        result = out if full.ndim > 1 else out[0]
        self._last_live_positions = [pos for pos in range(snap.ntotal) if pos not in snap.tombstones]
        return result

    def live_doc_ids(self):
        """Identities aligned with the columns of the last full-vector query result
        (ascending position order, tombstones excluded)."""
        return [self.pos2id[pos] for pos in getattr(self, '_last_live_positions', [])]

    @property
    def live_count(self):
        """Number of live (non-deleted) documents."""
        with self._lock:
            return len(self.id2pos)

    def vector_by_key(self, doc_id):
        """Get the current indexed vector for `doc_id` (post-replacement content)."""
        self._require_keyed()
        with self._lock:
            pos = self.id2pos[doc_id]  # KeyError on miss, on purpose
            replacement = self.replacements.get(pos)
        if replacement is not None:
            return replacement
        return self.vector_by_id(pos)

    def similarity_by_key(self, doc_id):
        """Get similarities of the document registered under `doc_id` against the index."""
        query = self.vector_by_key(doc_id)
        norm, self.norm = self.norm, False
        result = self[query]
        self.norm = norm
        return result

    def get_stats(self):
        """Snapshot of the keyed counters (adds / deletes / replaces / compactions /
        reclaim failures), for external consumption."""
        with self._lock:
            stats = dict(self.stats)
            stats['live'] = len(self.id2pos)
            stats['tombstones'] = len(self.tombstones)
            stats['round'] = self.round
        return stats

    #
    # compaction
    #

    def _maybe_compact(self):
        """Start a background compaction if the tombstone ratio warrants it."""
        if not self.auto_compact:
            return
        with self._lock:
            ntotal = len(self.fresh_docs) + sum(len(shard) for shard in self.shards)
            if ntotal == 0 or len(self.tombstones) < self.tombstone_ratio * ntotal:
                return
            if self._compact_thread is not None and self._compact_thread.is_alive():
                return
            self._compact_thread = threading.Thread(target=self._compact_safely, daemon=True)
        self._compact_thread.start()

    def _compact_safely(self):
        try:
            self.compact()
        except Exception:
            logger.exception("background compaction failed")

    def wait_for_compaction(self, timeout=None):
        """Block until a running background compaction finishes. Returns True if one ran."""
        thread = self._compact_thread
        if thread is None:
            return False
        thread.join(timeout)
        return not thread.is_alive()

    def _crash_hook(self, stage):
        """Test hook: simulate a power loss at a given compaction stage."""
        if self._compact_crash_at == stage:
            os._exit(1)

    def _write_journal(self, journal):
        fname = self._journal_fname()
        tmp = fname + '.tmp'
        with open(tmp, 'w') as fout:
            json.dump(journal, fout)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp, fname)

    def compact(self):
        """Reclaim tombstoned storage, synchronously.

        Live documents (with their replacements applied) are rewritten into new shard files
        next to the old ones; the swap is atomic and journal-based, so a crash leaves either
        the old or the new index, never a mixture. Queries against the old shards keep working
        throughout. Query results are identical before and after compaction.

        Returns
        -------
        bool
            True if anything was reclaimed.

        """
        self._require_keyed()
        with self._write_lock:
            self.close_shard()
            snap = self._snapshot()
            if not snap.tombstones and not snap.replacements:
                return False

            tmp_prefix = self.output_prefix + '.compact_tmp'
            sidecar_new = self._sidecar_fname() + '.new'
            old_files = [self.shardid2filename(i) for i in range(len(self.shards))]
            journal = {
                'stage': 'started',
                'tmp_prefix': tmp_prefix,
                'sidecar': self._sidecar_fname(),
                'sidecar_new': sidecar_new,
            }
            self._write_journal(journal)
            self._crash_hook('started')

            try:
                if self._fail_reclaim:
                    raise OSError(28, 'No space left on device (simulated)')
                live_positions = [pos for pos in range(snap.ntotal) if pos not in snap.tombstones]

                def live_corpus():
                    pos = 0
                    for shard in snap.shards:
                        matrix = shard.get_index().index
                        for row in matrix:
                            if pos in snap.replacements:
                                yield snap.replacements[pos]
                            elif pos not in snap.tombstones:
                                yield row
                            pos += 1

                tmp_index = Similarity(
                    tmp_prefix, live_corpus(), num_features=self.num_features, num_best=None,
                    chunksize=self.chunksize, shardsize=self.shardsize, norm=self.norm,
                )
                tmp_index.close_shard()

                new_pos2id = [snap.pos2id[pos] for pos in live_positions]
                new_id2pos = {doc_id: pos for pos, doc_id in enumerate(new_pos2id)}
                new_shards_meta = [
                    {
                        'fullname': self.shardid2filename(shard_no),
                        'length': len(shard),
                        'cls': shard.cls.__name__,
                        'num_nnz': getattr(shard, 'num_nnz', None),
                    }
                    for shard_no, shard in enumerate(tmp_index.shards)
                ]
                with self._lock:
                    self._state_version += 1
                    new_state = {
                        'format': 1,
                        'version': self._state_version,
                        'id2pos': new_id2pos,
                        'pos2id': new_pos2id,
                        'tombstones': set(),
                        'replacements': {},
                        'retired': self.retired,
                        'round': self.round + 1,
                        'stats': dict(self.stats, compactions=self.stats['compactions'] + 1),
                        'shards_meta': new_shards_meta,
                    }
                with open(sidecar_new, 'wb') as fout:
                    _stdlib_pickle.dump(new_state, fout, protocol=utils.PICKLE_PROTOCOL)
                    fout.flush()
                    os.fsync(fout.fileno())

                journal.update({
                    'stage': 'committed',
                    'renames': [
                        ('%s.%d' % (tmp_prefix, i), self.shardid2filename(i))
                        for i in range(len(tmp_index.shards))
                    ],
                    'old_shard_files': old_files,
                })
                self._write_journal(journal)
                self._crash_hook('committed')

                os.replace(sidecar_new, self._sidecar_fname())
                final_names = set()
                for tmp_name, final_name in journal['renames']:
                    os.replace(tmp_name, final_name)
                    final_names.add(final_name)
                for old in old_files:
                    if old not in final_names and os.path.exists(old):
                        os.remove(old)
                if os.path.exists(self._journal_fname()):
                    os.remove(self._journal_fname())

                # swap the in-memory state; in-flight queries keep their own snapshot
                for shard_no, shard in enumerate(tmp_index.shards):
                    shard.dirname, shard.fname = os.path.split(self.shardid2filename(shard_no))
                with self._lock:
                    self.shards = list(tmp_index.shards)
                    self.id2pos = new_id2pos
                    self.pos2id = new_pos2id
                    self.tombstones = set()
                    self.replacements = {}
                    self.round += 1
                    self.stats['compactions'] += 1
                self._crash_hook('swapped')
            except OSError as exc:
                with self._lock:
                    self.stats['reclaim_failures'] += 1
                    self._state_version += 1
                    self.last_reclaim_error = '%s: %s' % (type(exc).__name__, exc)
                    self._persist_locked()
                logger.error(
                    "reclaim failed, index left untouched (%d tombstones pending): %s",
                    len(snap.tombstones), self.last_reclaim_error,
                )
                self._rollback_compaction_files(journal)
                return False
        logger.info(
            "compaction complete: %d live identities, reclaimed %d positions in %d shards",
            len(new_pos2id), snap.ntotal - len(new_pos2id), len(old_files),
        )
        return True

    def _rollback_compaction_files(self, journal):
        """Remove partial compaction output (best effort)."""
        tmp_prefix = journal.get('tmp_prefix')
        if tmp_prefix:
            dirname = os.path.dirname(tmp_prefix) or '.'
            basename = os.path.basename(tmp_prefix)
            for fname in os.listdir(dirname):
                if fname.startswith(basename):
                    os.remove(os.path.join(dirname, fname))
        for tmp_name, _ in journal.get('renames', []):
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        sidecar_new = journal.get('sidecar_new')
        if sidecar_new and os.path.exists(sidecar_new):
            os.remove(sidecar_new)
        journal_fname = self._journal_fname()
        if os.path.exists(journal_fname):
            os.remove(journal_fname)

    def _recover_compaction(self):
        """Crash recovery: if a compaction journal exists, either roll the interrupted
        compaction back to the pre-compaction state or forward to the post-compaction state.
        Never leaves a mixture."""
        journal_fname = self._journal_fname()
        if not os.path.exists(journal_fname):
            return
        with open(journal_fname) as fin:
            journal = json.load(fin)
        if journal['stage'] == 'started':
            self._rollback_compaction_files(journal)
            logger.warning(
                "recovered from crash during compaction: rolled back to pre-compaction state",
            )
        else:  # 'committed': roll forward, every step is idempotent
            sidecar_new = journal['sidecar_new']
            if os.path.exists(sidecar_new):
                os.replace(sidecar_new, journal['sidecar'])
            final_names = set()
            for tmp_name, final_name in journal['renames']:
                if os.path.exists(tmp_name):
                    os.replace(tmp_name, final_name)
                final_names.add(final_name)
            for old in journal['old_shard_files']:
                if old not in final_names and os.path.exists(old):
                    os.remove(old)
            os.remove(journal_fname)
            logger.warning(
                "recovered from crash during compaction: completed interrupted compaction "
                "(%d shard files swapped in)", len(final_names),
            )
