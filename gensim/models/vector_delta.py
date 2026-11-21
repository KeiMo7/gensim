#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Gensim Contributors
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Compare two versions of a :class:`~gensim.models.keyedvectors.KeyedVectors` model,
and apply a delta-only incremental update between them.

Why this module exists
======================

A retrained word-vector model can change in two independent ways:

* the **vocabulary** changes (tokens are added or removed), and
* the **vectors** of the tokens present in both versions change.

Downstream similarity tasks react to both, and a full rebuild makes it hard to
tell which one is responsible.  This module adds an opt-in comparison layer on
top of the existing storage (:class:`~gensim.models.keyedvectors.KeyedVectors`)
and a delta-only update layer.  The training algorithms, the on-disk formats
and the existing load/save interfaces are untouched.

Quick start
===========

.. sourcecode:: pycon

    >>> from gensim.models.vector_delta import compare_versions, incremental_update
    >>> report = compare_versions('wiki-2025.kv', 'wiki-2026.kv', streaming=True)
    >>> report.write_json('wiki-diff.json')
    >>> report.write_markdown('wiki-diff.md')

    >>> record = incremental_update(
    ...     'wiki-2025.kv', 'wiki-2026.kv', 'wiki-2026.incremental.kv',
    ...     backup_dir='wiki-2025.backup', overwrite=True,
    ... )
    >>> record.write_json('wiki-update.json')

The command line front-end lives in :mod:`gensim.scripts.vector_delta`.

Key identity
============

Keys are compared after applying :class:`KeyNormalizer` (Unicode
normalization, whitespace handling and optionally case folding).  When two keys
of the *same* version normalize to the same string, the first occurrence in
source order wins -- the same rule the word2vec loaders use.  The number of
collisions is reported, never silently dropped.

Metrics and threshold
=====================

Two vectors are first checked for **bitwise equality**.  If their bytes differ,
the comparison reports:

* ``l2_delta`` -- Euclidean distance ``||v_new - v_old||``;
* ``relative_l2_delta`` -- ``l2_delta / ||v_old||`` (scale-aware);
* ``cosine_similarity`` / ``cosine_distance`` -- direction-aware.

A key counts as *changed* when ``relative_l2_delta > relative_l2_threshold``
**or** ``cosine_distance > cosine_distance_threshold`` (see
:class:`ChangePolicy`).  The defaults (``1e-3`` and ``1e-4``) are well above
float32 rounding noise (relative epsilon about ``1.2e-7``) and far below the
differences produced by a normal retrain.  Vectors that differ in the last bits
only are reported as ``unchanged_within_threshold`` so the grey zone is visible.

Streaming and memory
====================

``streaming=True`` never holds both vector matrices in memory.  The first
version is spilled to a temporary SQLite database (on disk), the second version
is streamed through it chunk by chunk, and only bounded-size statistics are
kept in RAM.  When the sources are gensim ``.kv`` files they are opened with
``mmap='r'``; word2vec text/binary files are read in fixed-size chunks.  Peak
RAM is therefore proportional to the chunk size, not the vocabulary size.

Known limitations are listed in the returned report and in the docstrings of
:func:`compare_versions` and :func:`incremental_update`.

"""

import datetime
import glob
import json
import logging
import math
import os
import shutil
import sqlite3
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors

logger = logging.getLogger(__name__)

__all__ = [
    'KeyNormalizer',
    'ChangePolicy',
    'DimensionMismatchError',
    'BaseVersionMissingError',
    'UnsupportedModelError',
    'ComparisonThresholdError',
    'KeyDelta',
    'VersionInfo',
    'VocabularySummary',
    'VectorSummary',
    'VersionComparisonReport',
    'UpdateRecord',
    'compare_versions',
    'explain_key',
    'incremental_update',
    'restore_backup',
    'evaluate_similarity_tasks',
    'compare_similarity_tasks',
    'iter_vectors',
]

SCHEMA_VERSION = 1
_EPS = 1e-12
_MAX_EXAMPLES_DEFAULT = 20
_SAMPLE_LIMIT_DEFAULT = 100_000
_CHUNK_DEFAULT = 10_000
_WORD2VEC_CHUNK_BYTES = 1024 * 1024
_SQLITE_IN_BATCH = 5_000
_INITIAL_DELTA_CAPACITY = 65_536


class DimensionMismatchError(ValueError):
    """Raised when two model versions cannot be compared because their vector sizes differ."""


class BaseVersionMissingError(FileNotFoundError):
    """Raised when a previous model version is required but is not available."""


class UnsupportedModelError(TypeError):
    """Raised for model types whose vector semantics make a plain vector delta misleading."""


class ComparisonThresholdError(ValueError):
    """Raised when comparison thresholds are invalid."""


# ---------------------------------------------------------------------------
# Key normalization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyNormalizer:
    """Define when two keys are considered the same word.

    The rules are applied in this order:

    1. Unicode normalization (``NFC`` by default; set to ``None`` to disable).
    2. Leading/trailing whitespace removal (``strip``).
    3. Runs of whitespace collapsed to a single space (``collapse_whitespace``).
    4. Unicode case folding when ``case_sensitive`` is ``False``.

    Non-string keys (for example integer Doc2Vec tags) are left unchanged: an
    integer identity is already unambiguous.

    Parameters
    ----------
    case_sensitive : bool, optional
        Keep the case of the key when True (default).  When False, ``str.casefold()``
        is used, which is more aggressive than ``str.lower()`` for scripts such as
        German ``ß`` or Greek final sigma.
    strip : bool, optional
        Remove leading and trailing whitespace (default True).
    collapse_whitespace : bool, optional
        Replace every internal run of whitespace (spaces, tabs, newlines, NBSP)
        with one ASCII space (default True).
    unicode_normalization : {None, 'NFC', 'NFD', 'NFKC', 'NFKD'}, optional
        Unicode normalization form.  Default ``'NFC'``; ``None`` disables it.

    Examples
    --------
    .. sourcecode:: pycon

        >>> KeyNormalizer().normalize('  New\\tYork ')
        'New York'
        >>> KeyNormalizer(case_sensitive=False).normalize('NEW york')
        'new york'

    """
    case_sensitive: bool = True
    strip: bool = True
    collapse_whitespace: bool = True
    unicode_normalization: Optional[str] = 'NFC'

    def __post_init__(self):
        if self.unicode_normalization not in (None, 'NFC', 'NFD', 'NFKC', 'NFKD'):
            raise ValueError(
                f"unsupported unicode_normalization {self.unicode_normalization!r}; "
                "use one of None, 'NFC', 'NFD', 'NFKC', 'NFKD'"
            )

    def normalize(self, key):
        """Return the normalized identity of ``key``."""
        if not isinstance(key, str):
            return key
        value = key
        if self.unicode_normalization:
            value = unicodedata.normalize(self.unicode_normalization, value)
        if self.strip:
            value = value.strip()
        if self.collapse_whitespace:
            value = ' '.join(value.split())
        if not self.case_sensitive:
            value = value.casefold()
        return value

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-serializable description, embedded in reports."""
        return {
            'case_sensitive': self.case_sensitive,
            'strip': self.strip,
            'collapse_whitespace': self.collapse_whitespace,
            'unicode_normalization': self.unicode_normalization,
            'rule': (
                'unicode_normalize -> strip -> collapse_whitespace'
                + (' -> casefold' if not self.case_sensitive else '')
            ),
            'non_string_keys': 'left unchanged',
            'duplicate_policy': 'first occurrence in source order wins; collisions are counted',
        }


# ---------------------------------------------------------------------------
# Change policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChangePolicy:
    """Decide which vector differences count as a real change.

    Parameters
    ----------
    cosine_distance_threshold : float, optional
        A changed direction is reported when ``1 - cos(v_old, v_new)`` exceeds this.
        Default ``1e-4`` (cosine similarity below ``0.9999``).
    relative_l2_threshold : float, optional
        A changed magnitude is reported when ``||v_new - v_old|| / ||v_old||``
        exceeds this.  Default ``1e-3`` (0.1% of the old length).
    sample_limit : int, optional
        Maximum number of delta values kept (reservoir-sampled, deterministic)
        for percentile reporting.  Default 100 000.
    sample_seed : int, optional
        Seed of the reservoir sampler.  Default 0, so reports are reproducible.

    Notes
    -----
    Bitwise-equal vectors are always classified as ``unchanged_exact`` and are
    never affected by the thresholds.  If the old vector has zero length the
    relative delta is reported as ``None`` and any non-zero difference counts
    as changed.

    """
    cosine_distance_threshold: float = 1e-4
    relative_l2_threshold: float = 1e-3
    sample_limit: int = _SAMPLE_LIMIT_DEFAULT
    sample_seed: int = 0

    def __post_init__(self):
        if not math.isfinite(self.cosine_distance_threshold) or self.cosine_distance_threshold < 0:
            raise ComparisonThresholdError('cosine_distance_threshold must be a finite value >= 0')
        if not math.isfinite(self.relative_l2_threshold) or self.relative_l2_threshold < 0:
            raise ComparisonThresholdError('relative_l2_threshold must be a finite value >= 0')
        if self.sample_limit < 0:
            raise ComparisonThresholdError('sample_limit must be >= 0')

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-serializable description, embedded in reports."""
        return {
            'cosine_distance_threshold': self.cosine_distance_threshold,
            'relative_l2_threshold': self.relative_l2_threshold,
            'exact_check': 'byte-for-byte equality of the stored vector',
            'changed_rule': (
                'not bitwise-equal AND (relative_l2_delta > relative_l2_threshold '
                'OR cosine_distance > cosine_distance_threshold)'
            ),
            'rationale': (
                'float32 relative epsilon is about 1.2e-7, so the default thresholds are '
                'roughly four orders of magnitude above representation noise, while a normal '
                'retrain moves vectors by far more.  Tune per model if needed.'
            ),
            'zero_vector_rule': (
                'if the old vector has zero length, relative_l2_delta is null and any '
                'non-zero difference counts as changed'
            ),
            'sample_limit': self.sample_limit,
            'sample_seed': self.sample_seed,
            'percentile_estimator': 'deterministic reservoir sample' if self.sample_limit else 'disabled',
        }


# ---------------------------------------------------------------------------
# Data classes used in reports
# ---------------------------------------------------------------------------

@dataclass
class VersionInfo:
    """Identity and shape of one model version."""
    label: str
    kind: str
    keys: int
    vector_size: int
    dtype: str
    derived_vectors: bool = False
    mmap: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VocabularySummary:
    """Vocabulary-level delta between two versions."""
    old_keys: int = 0
    new_keys: int = 0
    added: int = 0
    removed: int = 0
    common: int = 0
    changed: int = 0
    unchanged_exact: int = 0
    unchanged_within_threshold: int = 0
    old_normalization_collisions: int = 0
    new_normalization_collisions: int = 0
    added_examples: List[Any] = field(default_factory=list)
    removed_examples: List[Any] = field(default_factory=list)
    normalization_collision_examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VectorSummary:
    """Vector-level delta, restricted to keys present in both versions."""
    compared: int = 0
    changed: int = 0
    unchanged_exact: int = 0
    unchanged_within_threshold: int = 0
    mean_relative_l2_delta: Optional[float] = None
    max_relative_l2_delta: Optional[float] = None
    mean_cosine_distance: Optional[float] = None
    max_cosine_distance: Optional[float] = None
    relative_l2_percentiles: Optional[Dict[str, float]] = None
    cosine_distance_percentiles: Optional[Dict[str, float]] = None
    sampled: int = 0
    changed_examples: List[Dict[str, Any]] = field(default_factory=list)
    nearest_neighbor_examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KeyDelta:
    """Detailed, queryable delta for a single key."""
    key: Any
    normalized_key: Any
    status: str  # 'changed', 'unchanged_exact', 'unchanged_within_threshold', 'added', 'removed'
    vector_size: Optional[int] = None
    l2_delta: Optional[float] = None
    relative_l2_delta: Optional[float] = None
    cosine_similarity: Optional[float] = None
    cosine_distance: Optional[float] = None
    old_norm: Optional[float] = None
    new_norm: Optional[float] = None
    old_nearest_neighbors: List[Tuple[Any, float]] = field(default_factory=list)
    new_nearest_neighbors: List[Tuple[Any, float]] = field(default_factory=list)
    nearest_neighbor_changed: Optional[bool] = None
    nearest_neighbor_overlap: Optional[float] = None
    old_top1_rank_in_new: Optional[int] = None
    old_spelling: Optional[Any] = None
    new_spelling: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['old_nearest_neighbors'] = [[k, s] for k, s in self.old_nearest_neighbors]
        result['new_nearest_neighbors'] = [[k, s] for k, s in self.new_nearest_neighbors]
        return result


@dataclass
class VersionComparisonReport:
    """Result of :func:`compare_versions`.

    ``write_json`` and ``write_markdown`` serialize the report for humans and
    for downstream tooling (the JSON schema is versioned by ``schema_version``).
    """
    schema_version: int
    generated_at: str
    compatible: bool
    status: str
    reason: Optional[str]
    normalizer: Dict[str, Any]
    policy: Dict[str, Any]
    old: VersionInfo
    new: VersionInfo
    vocabulary: VocabularySummary
    vectors: VectorSummary
    streaming: bool
    similarity_tasks: Optional[Dict[str, Any]] = None
    limitations: List[str] = field(default_factory=list)

    @property
    def changed_keys(self) -> int:
        """Number of common keys whose vector changed beyond the policy threshold."""
        return self.vocabulary.changed

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema_version': self.schema_version,
            'generated_at': self.generated_at,
            'compatible': self.compatible,
            'status': self.status,
            'reason': self.reason,
            'normalizer': self.normalizer,
            'policy': self.policy,
            'old': self.old.to_dict(),
            'new': self.new.to_dict(),
            'vocabulary': self.vocabulary.to_dict(),
            'vectors': self.vectors.to_dict(),
            'streaming': self.streaming,
            'similarity_tasks': self.similarity_tasks,
            'limitations': list(self.limitations),
        }

    def write_json(self, path) -> None:
        """Write the report as UTF-8 JSON."""
        with utils.open(path, 'wb') as fout:
            fout.write(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True).encode('utf8'))

    def write_markdown(self, path) -> None:
        """Write a human-readable Markdown report."""
        data = self.to_dict()
        vocab, vectors, policy = data['vocabulary'], data['vectors'], data['policy']
        lines = [
            '# Word-vector version comparison',
            '',
            f'- Generated: {data["generated_at"]}',
            f'- Old: `{data["old"]["label"]}` ({data["old"]["keys"]} keys, dim {data["old"]["vector_size"]}, '
            f'{data["old"]["dtype"]})',
            f'- New: `{data["new"]["label"]}` ({data["new"]["keys"]} keys, dim {data["new"]["vector_size"]}, '
            f'{data["new"]["dtype"]})',
            f'- Status: **{data["status"]}**' + (f' -- {data["reason"]}' if data["reason"] else ''),
            f'- Streaming: {data["streaming"]}',
            '',
            '## Vocabulary',
            '',
            '| added | removed | common | changed vectors | unchanged (exact) | unchanged (within threshold) |',
            '|---:|---:|---:|---:|---:|---:|',
            f'| {vocab["added"]} | {vocab["removed"]} | {vocab["common"]} | {vocab["changed"]} | '
            f'{vocab["unchanged_exact"]} | {vocab["unchanged_within_threshold"]} |',
            '',
            f'- Old normalization collisions: {vocab["old_normalization_collisions"]}',
            f'- New normalization collisions: {vocab["new_normalization_collisions"]}',
        ]
        if vocab['added_examples']:
            lines.append(f'- Added examples: {vocab["added_examples"]}')
        if vocab['removed_examples']:
            lines.append(f'- Removed examples: {vocab["removed_examples"]}')
        lines += [
            '',
            '## Vector deltas',
            '',
            f'- Compared: {vectors["compared"]}',
            f'- Mean relative L2 delta: {vectors["mean_relative_l2_delta"]}',
            f'- Max relative L2 delta: {vectors["max_relative_l2_delta"]}',
            f'- Mean cosine distance: {vectors["mean_cosine_distance"]}',
            f'- Max cosine distance: {vectors["max_cosine_distance"]}',
            f'- Relative L2 percentiles: {vectors["relative_l2_percentiles"]}',
            f'- Cosine distance percentiles: {vectors["cosine_distance_percentiles"]}',
            '',
            '## Metric and thresholds',
            '',
            f'- Key identity: {data["normalizer"]["rule"]}',
            f'- {policy["changed_rule"]}',
            f'- Rationale: {policy["rationale"]}',
        ]
        if vectors['changed_examples']:
            lines += ['', '### Changed-key examples', '', '```json',
                      json.dumps(vectors['changed_examples'], indent=2, ensure_ascii=False, sort_keys=True),
                      '```']
        if vectors['nearest_neighbor_examples']:
            lines += ['', '### Nearest-neighbor examples', '', '```json',
                      json.dumps(
                          vectors['nearest_neighbor_examples'], indent=2, ensure_ascii=False, sort_keys=True),
                      '```']
        if data.get('similarity_tasks'):
            lines += ['', '## Similarity tasks', '', '```json',
                      json.dumps(data['similarity_tasks'], indent=2, ensure_ascii=False, sort_keys=True), '```']
        if data['limitations']:
            lines += ['', '## Limitations', ''] + [f'- {item}' for item in data['limitations']]
        with utils.open(path, 'wb') as fout:
            fout.write(('\n'.join(lines) + '\n').encode('utf8'))


@dataclass
class UpdateRecord:
    """Result of :func:`incremental_update`, including rollback information."""
    schema_version: int
    generated_at: str
    base: Dict[str, Any]
    updates: Dict[str, Any]
    output_path: str
    backup_dir: Optional[str]
    normalizer: Dict[str, Any]
    policy: Dict[str, Any]
    rows_total: int
    rows_reused_exact: int
    rows_replaced: int
    rows_added: int
    rows_unchanged_kept: int
    rows_pruned: int
    normalization_collisions: int
    changed_keys: List[Dict[str, Any]]
    elapsed_seconds: float
    completed: bool = True
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_json(self, path) -> None:
        """Write the update record as UTF-8 JSON."""
        with utils.open(path, 'wb') as fout:
            fout.write(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True).encode('utf8'))


# ---------------------------------------------------------------------------
# Source handling and streaming iteration
# ---------------------------------------------------------------------------

def _encode_identity(nkey):
    """Encode a normalized key for storage without str/int collisions."""
    if isinstance(nkey, str):
        return 's:' + nkey
    if isinstance(nkey, (int, np.integer)):
        return 'i:' + str(int(nkey))
    return 'o:' + repr(nkey)


def _is_pickle_file(path):
    try:
        with utils.open(path, 'rb') as fin:
            head = fin.read(2)
    except OSError:
        return False
    return head in (b'\x80\x04', b'\x80\x05')


def _detect_format(path, format='auto'):
    """Classify a path as 'kv', 'word2vec-text' or 'word2vec-binary'.

    Detection reads a 4 KiB sample after the header and decides:

    * a complete first record (``<word> <dim floats>``) followed by a newline is text;
    * a sample without a single newline whose completed tokens are all floats is a
      text record longer than the sample (long vectors);
    * anything else (including samples that are not valid UTF-8) is binary.

    Use ``format=...`` to override for unusual files.
    """
    path = os.fsdecode(os.fspath(path))
    if format != 'auto':
        if format not in ('kv', 'word2vec-text', 'word2vec-binary'):
            raise ValueError(
                f"unknown format {format!r}; use 'auto', 'kv', 'word2vec-text' or 'word2vec-binary'"
            )
        return format
    lower = path.lower()
    if lower.endswith(('.kv', '.model', '.pkl', '.pickle')) or _is_pickle_file(path):
        return 'kv'
    with utils.open(path, 'rb') as fin:
        header = fin.readline()
        try:
            dims = [int(x) for x in utils.to_unicode(header).split()]
        except (ValueError, UnicodeDecodeError):
            raise ValueError(
                f"{path!r} does not start with a '<count> <dim>' header and is not a gensim pickle; "
                "pass format='word2vec-text' / 'word2vec-binary' explicitly for header-less files"
            )
        if len(dims) != 2:
            raise ValueError(f"{path!r} has an invalid word2vec header: {header!r}")
        vector_size = dims[1]
        sample = fin.read(4096)
    try:
        decoded = sample.decode('utf8')
    except UnicodeDecodeError:
        return 'word2vec-binary'
    line, newline, _rest = decoded.partition('\n')
    parts = line.split(' ')
    if newline:
        if len(parts) == vector_size + 1:
            try:
                [float(x) for x in parts[1:]]
            except ValueError:
                return 'word2vec-binary'
            return 'word2vec-text'
        return 'word2vec-binary'
    # No newline in the sample.
    completed = parts[:-1]  # last token may be truncated by the sample boundary
    if len(completed) >= 2 and len(completed) <= vector_size + 1:
        try:
            [float(x) for x in completed[1:]]
        except ValueError:
            return 'word2vec-binary'
        return 'word2vec-text'
    return 'word2vec-binary'


def _as_source(source):
    """Return ``('object', kv)`` or ``('path', path)``."""
    if isinstance(source, KeyedVectors):
        return 'object', source
    if hasattr(source, 'wv') and isinstance(getattr(source, 'wv'), KeyedVectors):
        return 'object', source.wv  # full model: compare its word vectors
    if not isinstance(source, (str, bytes, os.PathLike)):
        raise TypeError(
            f"expected a KeyedVectors, a full model with .wv, or a path; got {type(source).__name__}"
        )
    path = os.fsdecode(os.fspath(source))
    return 'path', path


def _require_exists(path):
    if not os.path.exists(path):
        raise BaseVersionMissingError(
            f"model version {path!r} does not exist; a previous version is required for comparison or "
            "incremental update -- a full rebuild is the only option"
        )


def _load_kv(source, *, mmap=True):
    """Load a KeyedVectors from an object or a path."""
    kind, payload = _as_source(source)
    if kind == 'object':
        return payload
    _require_exists(payload)
    if not _detect_format(payload) == 'kv':
        return KeyedVectors.load_word2vec_format(
            payload, binary=_detect_format(payload) == 'word2vec-binary',
        )
    try:
        obj = KeyedVectors.load(payload, mmap='r' if mmap else None)
    except (IOError, ValueError) as err:
        logger.warning("could not mmap %s (%s); loading it fully into memory", payload, err)
        obj = KeyedVectors.load(payload, mmap=None)
    if hasattr(obj, 'wv') and isinstance(obj.wv, KeyedVectors):
        return obj.wv
    return obj


def _iter_word2vec_text(path, encoding, unicode_errors, chunk_size):
    with utils.open(path, 'rb') as fin:
        header = utils.to_unicode(fin.readline(), encoding=encoding).split()
        vector_size = int(header[1])
        keys, vectors = [], []
        for line in fin:
            parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(' ')
            if len(parts) < 2:
                continue
            keys.append(parts[0])
            vectors.append(np.asarray([float(x) for x in parts[1:1 + vector_size]], dtype=np.float32))
            if len(keys) >= chunk_size:
                yield keys, np.vstack(vectors)
                keys, vectors = [], []
        if keys:
            yield keys, np.vstack(vectors)


def _iter_word2vec_binary(path, encoding, unicode_errors, chunk_size,
                          binary_chunk_size=_WORD2VEC_CHUNK_BYTES):
    with utils.open(path, 'rb') as fin:
        header = utils.to_unicode(fin.readline(), encoding=encoding).split()
        vector_size = int(header[1])
        bytes_per_vector = vector_size * np.dtype(np.float32).itemsize
        buffer = b''
        keys, vectors = [], []
        while True:
            new_data = fin.read(binary_chunk_size)
            if not new_data:
                break
            buffer += new_data
            while True:
                space = buffer.find(b' ')
                if space < 0 or len(buffer) - space - 1 < bytes_per_vector:
                    break
                key = buffer[:space].decode(encoding, errors=unicode_errors)
                start = space + 1
                vector = np.frombuffer(buffer, dtype=np.float32, count=vector_size, offset=start).copy()
                buffer = buffer[start + bytes_per_vector:]
                keys.append(key)
                vectors.append(vector)
                if len(keys) >= chunk_size:
                    yield keys, np.vstack(vectors)
                    keys, vectors = [], []
            if len(new_data) < binary_chunk_size:
                break
        if buffer.strip():
            raise EOFError(f"unexpected trailing data in binary word2vec file {path!r}")
        if keys:
            yield keys, np.vstack(vectors)


def _iter_resolved(resolved, *, chunk_size) -> Iterator[Tuple[List[Any], np.ndarray]]:
    """Iterate chunks over an already-resolved source tuple."""
    kind = resolved[0]
    if kind == 'object':
        kv = resolved[1]
        for start in range(0, len(kv.index_to_key), chunk_size):
            stop = min(start + chunk_size, len(kv.index_to_key))
            yield list(kv.index_to_key[start:stop]), kv.vectors[start:stop]
        return
    _kind, resolved_format, path, _label = resolved
    if path.lower().endswith(('.gz', '.bz2')):
        raise UnsupportedModelError(
            f"{path!r} is compressed; streaming iteration cannot seek it. "
            "Decompress it first, or pass a KeyedVectors instance."
        )
    if resolved_format == 'word2vec-binary':
        yield from _iter_word2vec_binary(path, 'utf8', 'strict', chunk_size)
    else:
        yield from _iter_word2vec_text(path, 'utf8', 'strict', chunk_size)


def _resolve_source(source, format='auto'):
    """Resolve a source once: ``('object', kv, label)`` or ``('file', fmt, path, label)``.

    ``.kv``/``.model`` paths are loaded exactly once (mmap) here, so metadata
    inspection and iteration share the same arrays instead of loading twice.
    """
    kind, payload = _as_source(source)
    if kind == 'object':
        return ('object', payload, f'{payload.__class__.__name__} at 0x{id(payload):x}')
    _require_exists(payload)
    resolved_format = _detect_format(payload, format)
    if resolved_format == 'kv':
        return ('object', _load_kv(payload, mmap=True), payload)
    return ('file', resolved_format, payload, payload)


def _resolved_info(resolved) -> Tuple[VersionInfo, List[str]]:
    """Describe a resolved source without materializing its vectors."""
    limitations = []
    kind = resolved[0]
    if kind == 'object':
        kv, label = resolved[1], resolved[2]
        is_ft = kv.__class__.__name__ == 'FastTextKeyedVectors'
        info = VersionInfo(
            label=label,
            kind=kv.__class__.__name__,
            keys=len(kv),
            vector_size=kv.vector_size,
            dtype=str(kv.vectors.dtype),
            derived_vectors=is_ft,
            mmap=isinstance(kv.vectors, np.memmap),
        )
        if is_ft:
            limitations.append(
                'FastTextKeyedVectors: comparison uses the derived full-word vectors; '
                'subword/ngram vectors are not compared.'
            )
        if kv.__class__.__name__ == 'PoincareKeyedVectors':
            limitations.append(
                'PoincareKeyedVectors: downstream distance is hyperbolic; cosine/L2 deltas are advisory only.'
            )
        return info, limitations
    _kind, resolved_format, path, label = resolved
    with utils.open(path, 'rb') as fin:
        header = utils.to_unicode(fin.readline()).split()
        keys, vector_size = int(header[0]), int(header[1])
    info = VersionInfo(
        label=label,
        kind=resolved_format,
        keys=keys,
        vector_size=vector_size,
        dtype='float32',
        derived_vectors=False,
        mmap=False,
    )
    limitations.append(
        'word2vec-format input: vocabulary counts/expandos are not persisted, so only vector data is compared.'
    )
    return info, limitations


def iter_vectors(
        source, *, format='auto', chunk_size=_CHUNK_DEFAULT, encoding='utf8', unicode_errors='strict',
    ) -> Iterator[Tuple[List[Any], np.ndarray]]:
    """Yield ``(keys, vectors)`` chunks from a model or a vector file.

    ``source`` may be a :class:`~gensim.models.keyedvectors.KeyedVectors`
    instance (or a full model exposing ``.wv``), a gensim-saved ``.kv`` /
    ``.model`` file (memory-mapped when possible), or a word2vec text/binary
    file.  Only ``chunk_size`` vectors are materialized at a time.

    ``format`` is ``'auto'`` (default), ``'kv'``, ``'word2vec-text'`` or
    ``'word2vec-binary'``.

    Notes
    -----
    Compressed gensim files are loaded fully by ``KeyedVectors.load`` (mmap is
    impossible there); for million-scale comparison prefer an uncompressed
    ``.kv`` file or a word2vec-format file.

    """
    yield from _iter_resolved(_resolve_source(source, format), chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Streaming stores
# ---------------------------------------------------------------------------

class _MemoryStore:
    """Bounded by vocabulary size; used when streaming is disabled."""

    def __init__(self):
        self._rows = {}
        self.collisions = 0
        self.collision_examples = []
        self.new_collisions = 0
        self._seen_new = set()

    def put_many(self, tokens, keys, vectors):
        for token, key, vector in zip(tokens, keys, vectors):
            if token in self._rows:
                self.collisions += 1
                if len(self.collision_examples) < 5:
                    self.collision_examples.append(
                        {'normalized_key': key, 'kept': self._rows[token][0], 'ignored': key}
                    )
                continue
            self._rows[token] = (key, np.array(vector, copy=True))

    def pop_many(self, tokens):
        result = []
        for token in tokens:
            row = self._rows.pop(token, None)
            if row is None:
                result.append(None)
            else:
                key, vector = row
                result.append((key, str(vector.dtype), vector.tobytes()))
        return result

    def remaining(self):
        return len(self._rows)

    def remaining_examples(self, limit):
        return [key for key, _ in list(self._rows.values())[:limit]]

    def close(self):
        pass

    def filter_new(self, tokens):
        """First occurrence per identity, tracked in memory."""
        fresh = []
        for token in tokens:
            if token in self._seen_new:
                self.new_collisions += 1
                continue
            self._seen_new.add(token)
            fresh.append(token)
        return fresh


class _SqliteStore:
    """Disk-backed spill store: RAM use is independent of the vocabulary size."""

    def __init__(self):
        fd, path = tempfile.mkstemp(prefix='gensim-vector-delta-', suffix='.sqlite3')
        os.close(fd)
        self.path = path
        self.collisions = 0
        self.collision_examples = []
        self.new_collisions = 0
        self._conn = sqlite3.connect(path)
        self._conn.execute('PRAGMA journal_mode=OFF')
        self._conn.execute('PRAGMA synchronous=OFF')
        self._conn.execute('PRAGMA temp_store=FILE')
        self._conn.execute('CREATE TABLE old_vecs (nkey TEXT PRIMARY KEY, key BLOB, dtype TEXT, vec BLOB)')
        self._conn.execute('CREATE TABLE seen_new (nkey TEXT PRIMARY KEY)')

    def _existing(self, tokens, table='old_vecs'):
        if not tokens:
            return set()
        found = set()
        for start in range(0, len(tokens), _SQLITE_IN_BATCH):
            batch = tokens[start:start + _SQLITE_IN_BATCH]
            placeholders = ','.join('?' * len(batch))
            rows = self._conn.execute(
                f'SELECT nkey FROM {table} WHERE nkey IN ({placeholders})', batch
            ).fetchall()
            found.update(row[0] for row in rows)
        return found

    def filter_new(self, tokens):
        """Return the first occurrence of each token (dedupe across the whole new side).

        Uses a disk table, so memory stays bounded for million-key vocabularies.
        """
        if not tokens:
            return []
        already = self._existing(tokens, 'seen_new')
        fresh = []
        within = set()
        for token in tokens:
            if token in already or token in within:
                self.new_collisions += 1
                continue
            within.add(token)
            fresh.append(token)
        if fresh:
            self._conn.executemany('INSERT OR IGNORE INTO seen_new (nkey) VALUES (?)', [(t,) for t in fresh])
            self._conn.commit()
        return fresh

    def put_many(self, tokens, keys, vectors):
        if not tokens:
            return
        existing = self._existing(tokens)
        retained = {}
        if existing:
            for start in range(0, len(tokens), _SQLITE_IN_BATCH):
                batch = tokens[start:start + _SQLITE_IN_BATCH]
                placeholders = ','.join('?' * len(batch))
                for token, key in self._conn.execute(
                    f'SELECT nkey, key FROM old_vecs WHERE nkey IN ({placeholders})', batch
                ):
                    retained[token] = _unpack_key(key)
        rows = []
        within = {}
        for token, key, vector in zip(tokens, keys, vectors):
            if token in within or token in existing:
                self.collisions += 1
                if len(self.collision_examples) < 5:
                    kept = within[token] if token in within else retained.get(token)
                    self.collision_examples.append(
                        {'normalized_key': key, 'kept': kept, 'ignored': key}
                    )
                continue
            within[token] = key
            rows.append((token, _pack_key(key), str(vector.dtype), vector.tobytes()))
        self._conn.executemany('INSERT OR IGNORE INTO old_vecs (nkey, key, dtype, vec) VALUES (?, ?, ?, ?)', rows)
        self._conn.commit()

    def pop_many(self, tokens):
        if not tokens:
            return []
        found = {}
        for start in range(0, len(tokens), _SQLITE_IN_BATCH):
            batch = tokens[start:start + _SQLITE_IN_BATCH]
            placeholders = ','.join('?' * len(batch))
            rows = self._conn.execute(
                f'SELECT nkey, key, dtype, vec FROM old_vecs WHERE nkey IN ({placeholders})', batch
            ).fetchall()
            for token, key, dtype, vec in rows:
                found[token] = (key, dtype, vec)
        result = []
        for token in tokens:
            row = found.pop(token, None)
            result.append(None if row is None else (_unpack_key(row[0]), row[1], row[2]))
        for start in range(0, len(tokens), _SQLITE_IN_BATCH):
            batch = tokens[start:start + _SQLITE_IN_BATCH]
            placeholders = ','.join('?' * len(batch))
            self._conn.execute(f'DELETE FROM old_vecs WHERE nkey IN ({placeholders})', batch)
        self._conn.commit()
        return result

    def remaining(self):
        return self._conn.execute('SELECT COUNT(*) FROM old_vecs').fetchone()[0]

    def remaining_examples(self, limit):
        rows = self._conn.execute('SELECT key FROM old_vecs ORDER BY rowid LIMIT ?', (limit,)).fetchall()
        return [_unpack_key(row[0]) for row in rows]

    def close(self):
        try:
            self._conn.close()
        finally:
            for suffix in ('', '-journal', '-wal', '-shm'):
                try:
                    os.unlink(self.path + suffix)
                except OSError:
                    pass


def _pack_key(key):
    """Pack a key for SQLite (strings as UTF-8, other types as JSON)."""
    if isinstance(key, str):
        return b's' + key.encode('utf8')
    return b'j' + json.dumps(key, ensure_ascii=False).encode('utf8')


def _unpack_key(blob):
    if blob[:1] == b's':
        return blob[1:].decode('utf8')
    return json.loads(blob[1:].decode('utf8'))


# ---------------------------------------------------------------------------
# Delta arithmetic and statistics
# ---------------------------------------------------------------------------

def _batch_delta(old_matrix, new_matrix, policy: ChangePolicy):
    """Vectorized metric bundle for a batch of aligned vector rows.

    Returns a dict of numpy arrays: ``exact``, ``status_code`` (0 exact,
    1 changed, 2 within), ``l2_delta``, ``relative_l2_delta`` (NaN when
    undefined), ``cosine_distance`` (NaN when undefined), ``old_norm``,
    ``new_norm``.
    """
    old_matrix = np.asarray(old_matrix)
    new_matrix = np.asarray(new_matrix)
    n = old_matrix.shape[0]
    if old_matrix.dtype == new_matrix.dtype:
        old_bytes = np.ascontiguousarray(old_matrix).view(np.uint8).reshape(n, -1)
        new_bytes = np.ascontiguousarray(new_matrix).view(np.uint8).reshape(n, -1)
        exact = np.all(old_bytes == new_bytes, axis=1)
    else:
        exact = np.zeros(n, dtype=bool)
    old64 = old_matrix.astype(np.float64)
    new64 = new_matrix.astype(np.float64)
    diff = new64 - old64
    l2 = np.linalg.norm(diff, axis=1)
    old_norm = np.linalg.norm(old64, axis=1)
    new_norm = np.linalg.norm(new64, axis=1)
    relative = np.full(n, np.nan)
    nonzero_old = old_norm > _EPS
    relative[nonzero_old] = l2[nonzero_old] / old_norm[nonzero_old]
    relative[(~nonzero_old) & (l2 == 0.0)] = 0.0
    cosine = np.full(n, np.nan)
    usable = nonzero_old & (new_norm > _EPS)
    if usable.any():
        dots = np.einsum('ij,ij->i', old64[usable], new64[usable])
        cos_sim = np.clip(dots / (old_norm[usable] * new_norm[usable]), -1.0, 1.0)
        cosine[usable] = 1.0 - cos_sim
    changed = np.zeros(n, dtype=bool)
    rel_over = nonzero_old & (relative > policy.relative_l2_threshold)
    changed |= rel_over
    changed |= (~nonzero_old) & (l2 > 0.0)
    cos_over = usable & (cosine > policy.cosine_distance_threshold)
    changed |= cos_over
    status = np.full(n, 1, dtype=np.int8)
    status[exact] = 0
    status[~exact & ~changed] = 2
    return {
        'exact': exact,
        'status': status,
        'l2_delta': l2,
        'relative_l2_delta': relative,
        'cosine_distance': cosine,
        'old_norm': old_norm,
        'new_norm': new_norm,
    }


def _single_delta(old_vec, new_vec, policy: ChangePolicy) -> Dict[str, Any]:
    """Compute one key pair's metrics (thin wrapper over :func:`_batch_delta`)."""
    batch = _batch_delta(
        np.asarray(old_vec).reshape(1, -1), np.asarray(new_vec).reshape(1, -1), policy,
    )
    status = int(batch['status'][0])
    relative = float(batch['relative_l2_delta'][0])
    cosine = float(batch['cosine_distance'][0])
    old_norm = float(batch['old_norm'][0])
    return {
        'status': {0: 'unchanged_exact', 1: 'changed', 2: 'unchanged_within_threshold'}[status],
        'l2_delta': float(batch['l2_delta'][0]),
        'relative_l2_delta': None if math.isnan(relative) else relative,
        'cosine_similarity': None if math.isnan(cosine) else 1.0 - cosine,
        'cosine_distance': None if math.isnan(cosine) else cosine,
        'old_norm': old_norm,
        'new_norm': float(batch['new_norm'][0]),
    }


class _Accumulator:
    """Exact counts plus bounded-size summary statistics."""

    def __init__(self, policy: ChangePolicy):
        self.policy = policy
        self.compared = 0
        self.changed = 0
        self.exact = 0
        self.within = 0
        self._sum_rel = 0.0
        self._count_rel = 0
        self._max_rel = None
        self._sum_cos = 0.0
        self._count_cos = 0
        self._max_cos = None
        limit = policy.sample_limit
        self._rng = np.random.Generator(np.random.PCG64(policy.sample_seed))
        self._rel_sample = np.empty(limit, dtype=np.float64) if limit else None
        self._cos_sample = np.empty(limit, dtype=np.float64) if limit else None
        self._rel_filled = 0
        self._cos_filled = 0

    def add_batch(self, batch):
        status = batch['status']
        n = len(status)
        self.compared += n
        self.exact += int(np.count_nonzero(status == 0))
        self.changed += int(np.count_nonzero(status == 1))
        self.within += int(np.count_nonzero(status == 2))
        relative = batch['relative_l2_delta']
        finite_rel = np.isfinite(relative)
        if finite_rel.any():
            values = relative[finite_rel]
            self._sum_rel += float(values.sum())
            self._count_rel += int(values.size)
            chunk_max = float(values.max())
            self._max_rel = chunk_max if self._max_rel is None else max(self._max_rel, chunk_max)
            self._rel_filled = self._reservoir(self._rel_sample, self._rel_filled, values)
        cosine = batch['cosine_distance']
        finite_cos = np.isfinite(cosine)
        if finite_cos.any():
            values = cosine[finite_cos]
            self._sum_cos += float(values.sum())
            self._count_cos += int(values.size)
            chunk_max = float(values.max())
            self._max_cos = chunk_max if self._max_cos is None else max(self._max_cos, chunk_max)
            self._cos_filled = self._reservoir(self._cos_sample, self._cos_filled, values)

    def _reservoir(self, reservoir, filled, values):
        """Vectorized reservoir sampling (Algorithm R), deterministic per seed."""
        if reservoir is None:
            return filled
        limit = len(reservoir)
        start = filled
        if start < limit:
            take = min(limit - start, len(values))
            reservoir[start:start + take] = values[:take]
            filled += take
            values = values[take:]
            start = limit
        if len(values) == 0:
            return filled
        # Global indices of remaining values are start..end-1 (0-based).
        global_index = np.arange(start, start + len(values), dtype=np.int64)
        probabilities = limit / (global_index + 1.0)
        draws = self._rng.random(len(values))
        accept = draws < probabilities
        if accept.any():
            slots = self._rng.integers(0, limit, size=int(np.count_nonzero(accept)))
            reservoir[slots] = values[accept]
        return filled

    @staticmethod
    def _percentiles(values, filled):
        if values is None or filled == 0:
            return None
        arr = values[:filled]
        return {f'p{p}': float(np.percentile(arr, p)) for p in (50, 90, 95, 99, 100)}

    def summary(self) -> Dict[str, Any]:
        return {
            'compared': self.compared,
            'changed': self.changed,
            'unchanged_exact': self.exact,
            'unchanged_within_threshold': self.within,
            'mean_relative_l2_delta': self._sum_rel / self._count_rel if self._count_rel else None,
            'max_relative_l2_delta': self._max_rel,
            'mean_cosine_distance': self._sum_cos / self._count_cos if self._count_cos else None,
            'max_cosine_distance': self._max_cos,
            'relative_l2_percentiles': self._percentiles(self._rel_sample, self._rel_filled),
            'cosine_distance_percentiles': self._percentiles(self._cos_sample, self._cos_filled),
            'sampled': self._rel_filled,
        }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _validate_comparable(old_info: VersionInfo, new_info: VersionInfo):
    if old_info.vector_size != new_info.vector_size:
        raise DimensionMismatchError(
            'refusing to compare models with different vector sizes: '
            f"old {old_info.label!r} has dim {old_info.vector_size}, "
            f"new {new_info.label!r} has dim {new_info.vector_size}. "
            'A dimension change means the vector spaces are not comparable; rebuild the downstream '
            'index/model from the new version instead of comparing or incrementally updating.'
        )
    if old_info.derived_vectors or new_info.derived_vectors:
        logger.warning("comparing derived fastText vectors; subword vectors themselves are not compared")
    if 'PoincareKeyedVectors' in (old_info.kind, new_info.kind):
        logger.warning("PoincareKeyedVectors use hyperbolic distance; cosine/L2 deltas are advisory only")


def _compare_streams(
        old_chunks, make_new_chunks, *, normalizer, policy, store, max_examples, release_old=None,
    ) -> Tuple[VocabularySummary, Dict[str, Any]]:
    """Two-pass streaming comparison: spill old, release it, stream new through the store."""
    vocab = VocabularySummary()
    acc = _Accumulator(policy)
    changed_examples: List[Dict[str, Any]] = []
    added_examples: List[Any] = []

    old_raw = 0
    for keys, vectors in old_chunks:
        if len(keys) != len(vectors):
            raise ValueError(f'inconsistent chunk: {len(keys)} keys vs {len(vectors)} vectors')
        old_raw += len(keys)
        tokens = [_encode_identity(normalizer.normalize(key)) for key in keys]
        store.put_many(tokens, keys, vectors)
    old_unique = store.remaining()
    vocab.old_keys = old_raw
    vocab.old_normalization_collisions = old_raw - old_unique
    vocab.normalization_collision_examples.extend(store.collision_examples[:5])

    # The old model has served its purpose.  Releasing it here (a callback because
    # the caller owns the reference) lets the OS reclaim its mmap pages before the
    # new model's pages are touched, so streaming peak RAM stays near one model.
    del old_chunks
    if release_old is not None:
        release_old()
    new_chunks = make_new_chunks()

    new_raw = 0
    for keys, vectors in new_chunks:
        if len(keys) != len(vectors):
            raise ValueError(f'inconsistent chunk: {len(keys)} keys vs {len(vectors)} vectors')
        new_raw += len(keys)
        tokens = [_encode_identity(normalizer.normalize(key)) for key in keys]
        fresh = set(store.filter_new(tokens))
        unique_tokens, unique_keys, unique_vectors = [], [], []
        for token, key, vector in zip(tokens, keys, vectors):
            if token not in fresh:
                continue
            fresh.discard(token)  # keep the first occurrence within this chunk
            unique_tokens.append(token)
            unique_keys.append(key)
            unique_vectors.append(vector)
        previous = store.pop_many(unique_tokens)

        matched_keys, old_rows, new_rows = [], [], []
        for key, vector, prior in zip(unique_keys, unique_vectors, previous):
            if prior is None:
                if len(added_examples) < max_examples:
                    added_examples.append(key)
                continue
            _old_key, old_dtype, old_bytes = prior
            matched_keys.append(key)
            old_rows.append(np.frombuffer(old_bytes, dtype=np.dtype(old_dtype)))
            new_rows.append(vector)
        vocab.added += len(unique_tokens) - len(matched_keys)
        if matched_keys:
            batch = _batch_delta(np.vstack(old_rows), np.vstack(new_rows), policy)
            acc.add_batch(batch)
            vocab.common += len(matched_keys)
            status = batch['status']
            vocab.changed += int(np.count_nonzero(status == 1))
            vocab.unchanged_exact += int(np.count_nonzero(status == 0))
            vocab.unchanged_within_threshold += int(np.count_nonzero(status == 2))
            for local, key in enumerate(matched_keys):
                if status[local] != 1 or len(changed_examples) >= max_examples:
                    continue
                cosine = float(batch['cosine_distance'][local])
                relative = float(batch['relative_l2_delta'][local])
                changed_examples.append({
                    'key': key,
                    'l2_delta': float(batch['l2_delta'][local]),
                    'relative_l2_delta': None if math.isnan(relative) else relative,
                    'cosine_distance': None if math.isnan(cosine) else cosine,
                })
    vocab.new_keys = new_raw
    vocab.new_normalization_collisions = getattr(store, 'new_collisions', 0)
    vocab.removed = store.remaining()
    vocab.removed_examples = store.remaining_examples(max_examples)
    vocab.added_examples = added_examples
    if store.collision_examples and not vocab.normalization_collision_examples:
        vocab.normalization_collision_examples = store.collision_examples[:5]
    summary = acc.summary()
    summary['changed_examples'] = changed_examples
    return vocab, summary


def _neighbor_examples(old, new, changed_examples, normalizer, topn) -> List[Dict[str, Any]]:
    """Old/new nearest neighbours for a few changed keys (opt-in)."""
    if not changed_examples or topn <= 0:
        return []
    old_kv = _load_kv(old, mmap=True)
    new_kv = _load_kv(new, mmap=True)
    examples = []
    for example in changed_examples:
        delta = explain_key(
            old_kv, new_kv, example['key'], normalizer=normalizer, topn=topn, _preloaded=True,
        )
        examples.append(delta.to_dict())
    return examples


def compare_versions(
        old, new, *, normalizer=None, policy=None, streaming=None, format='auto',
        chunk_size=_CHUNK_DEFAULT, max_examples=_MAX_EXAMPLES_DEFAULT, nearest_neighbor_examples=0,
        nearest_neighbor_topn=10, similarity_tasks=None,
    ) -> VersionComparisonReport:
    """Compare two versions of a word-vector model.

    Parameters
    ----------
    old, new : KeyedVectors or str or path-like
        The two versions.  Each may be an in-memory
        :class:`~gensim.models.keyedvectors.KeyedVectors` (a full model exposing
        ``.wv`` is accepted too), a gensim ``.kv`` / ``.model`` file, or a
        word2vec text/binary file.
    normalizer : KeyNormalizer, optional
        Key identity rules.  Defaults to :class:`KeyNormalizer` (exact case,
        stripped/collapsed whitespace, NFC).
    policy : ChangePolicy, optional
        Change metric and thresholds.  Defaults to :class:`ChangePolicy`.
    streaming : bool, optional
        ``True`` spills the old vectors to a temporary SQLite database and
        keeps only chunk-sized data in RAM.  ``None`` (default) enables
        streaming automatically when either source is a path.
    format : str, optional
        Source format for paths: ``'auto'``, ``'kv'``, ``'word2vec-text'`` or
        ``'word2vec-binary'``.
    chunk_size : int, optional
        Vectors processed per chunk (default 10 000).
    max_examples : int, optional
        How many added/removed/changed key examples to embed in the report.
    nearest_neighbor_examples : int, optional
        If > 0, compute old/new nearest neighbours for that many changed keys.
        Requires loading both models (mmap is used for paths).
    nearest_neighbor_topn : int, optional
        Neighbour list size for the examples (default 10).
    similarity_tasks : dict, optional
        When given, evaluate both versions on similarity tasks and embed the
        before/after comparison.  Supported keys: ``pairs`` (path for
        :meth:`KeyedVectors.evaluate_word_pairs`), ``analogies`` (path for
        :meth:`KeyedVectors.evaluate_word_analogies`), and optional
        ``pairs_kwargs`` / ``analogies_kwargs`` dicts.

    Returns
    -------
    VersionComparisonReport
        Counts, metric/threshold descriptions, examples and optional
        similarity-task results.  Use ``write_json`` / ``write_markdown``.

    Raises
    ------
    DimensionMismatchError
        If the two versions have different vector sizes.  Published under its
        own name (a :class:`ValueError`) so callers can distinguish it from
        other input errors.
    BaseVersionMissingError
        If a path does not exist (``FileNotFoundError``).
    UnsupportedModelError
        For compressed streaming input that cannot be read incrementally.

    Notes
    -----
    Streaming uses a temporary SQLite file whose size is proportional to the
    old version's vectors; it is deleted on exit.  The comparison is
    single-threaded and deterministic: source order decides which duplicate
    normalized key wins, and the percentile sample uses a fixed seed.

    """
    normalizer = normalizer or KeyNormalizer()
    policy = policy or ChangePolicy()
    if streaming is None:
        streaming = not (isinstance(old, KeyedVectors) and isinstance(new, KeyedVectors))

    old_resolved = _resolve_source(old, format)
    new_resolved = _resolve_source(new, format)
    old_info, old_limitations = _resolved_info(old_resolved)
    new_info, new_limitations = _resolved_info(new_resolved)
    limitations = list(dict.fromkeys(old_limitations + new_limitations))
    limitations.extend([
        'comparison does not measure downstream task quality by itself; pass similarity_tasks or '
        'call compare_similarity_tasks for that.',
        'keys are matched by normalized identity; per-key attributes such as count are not compared.',
        'no concurrency is introduced: the comparison is single-threaded and deterministic; '
        'do not modify the source files while it runs.',
    ])

    report = VersionComparisonReport(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        compatible=True,
        status='compared',
        reason=None,
        normalizer=normalizer.describe(),
        policy=policy.describe(),
        old=old_info,
        new=new_info,
        vocabulary=VocabularySummary(),
        vectors=VectorSummary(),
        streaming=bool(streaming),
        limitations=limitations,
    )

    _validate_comparable(old_info, new_info)

    store = _SqliteStore() if streaming else _MemoryStore()
    if old_resolved[0] == 'object' and not isinstance(old, KeyedVectors):
        # We loaded this path ourselves; holding a second reference here would keep
        # the old model's pages alive while the new model is streamed in.
        old_holder = list(old_resolved)
    else:
        old_holder = None

    def release_old():
        # The old model has been fully spilled to the store; drop our own reference
        # so its mmap pages can be reclaimed before the new model is touched.
        nonlocal old_resolved
        if old_holder is not None:
            old_holder[1] = None
            old_resolved = [None, None, old_resolved[2]]

    try:
        vocab, vector_summary = _compare_streams(
            _iter_resolved(old_resolved, chunk_size=chunk_size),
            lambda: _iter_resolved(new_resolved, chunk_size=chunk_size),
            normalizer=normalizer, policy=policy, store=store, max_examples=max_examples,
            release_old=release_old,
        )
        report.vocabulary = vocab
        report.vectors = VectorSummary(**vector_summary)
    finally:
        store.close()

    if nearest_neighbor_examples > 0:
        report.vectors.nearest_neighbor_examples = _neighbor_examples(
            old, new, report.vectors.changed_examples[:nearest_neighbor_examples],
            normalizer, nearest_neighbor_topn,
        )

    if similarity_tasks:
        report.similarity_tasks = compare_similarity_tasks(old, new, **similarity_tasks)

    return report


# ---------------------------------------------------------------------------
# Per-key query
# ---------------------------------------------------------------------------

def _find_index(kv: KeyedVectors, key, normalizer: KeyNormalizer):
    """Find a key's index under normalized identity (fast path for exact spelling)."""
    if isinstance(key, str) and key in kv.key_to_index:
        return kv.key_to_index[key], key
    nkey = normalizer.normalize(key)
    for index, candidate in enumerate(kv.index_to_key):
        if normalizer.normalize(candidate) == nkey:
            return index, candidate
    return None, None


def _neighbors(kv: KeyedVectors, vector, topn, exclude_index=None):
    if topn <= 0 or not len(kv):
        return []
    query = np.asarray(vector, dtype=np.float64)
    query_norm = np.linalg.norm(query)
    if query_norm <= _EPS:
        return []
    matrix = np.asarray(kv.vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sims = (matrix @ query) / (norms * query_norm)
    sims = np.nan_to_num(sims, nan=-np.inf)
    order = np.argsort(-sims, kind='stable')
    result = []
    for index in order:
        if index == exclude_index or norms[index] <= _EPS:
            continue
        result.append((kv.index_to_key[index], float(sims[index])))
        if len(result) >= topn:
            break
    return result


def explain_key(
        old, new, key, *, normalizer=None, policy=None, topn=10, _preloaded=False,
    ) -> KeyDelta:
    """Explain what happened to one key between two versions.

    Parameters
    ----------
    old, new : KeyedVectors or str or path-like
        The two versions (in-memory objects or files; files are mmap-loaded).
    key : str or int
        The key to look up, under the identity rules of ``normalizer``.
    normalizer : KeyNormalizer, optional
        Key identity rules.  Defaults to :class:`KeyNormalizer`.
    policy : ChangePolicy, optional
        Change metric and thresholds.
    topn : int, optional
        Number of nearest neighbours to report for each version (default 10).

    Returns
    -------
    KeyDelta
        Offset metrics and whether the nearest neighbour changed.  ``status``
        is ``changed``, ``unchanged_exact``, ``unchanged_within_threshold``,
        ``added`` or ``removed``.  ``old_spelling``/``new_spelling`` show the
        stored spelling when it differs from the query.

    Notes
    -----
    Looking up a key by a spelling that differs from the stored spelling under
    normalization requires a linear scan of the vocabulary; lookups by the
    stored spelling use the normal hash index.  Neighbour computation is
    vectorized over the full vocabulary and uses float64 scratch (~2x the
    model's vector bytes), which is fine for interactive queries but not for
    million-scale batch loops.

    """
    normalizer = normalizer or KeyNormalizer()
    policy = policy or ChangePolicy()
    old_kv = old if _preloaded and isinstance(old, KeyedVectors) else _load_kv(old, mmap=True)
    new_kv = new if _preloaded and isinstance(new, KeyedVectors) else _load_kv(new, mmap=True)
    if old_kv.vector_size != new_kv.vector_size:
        raise DimensionMismatchError(
            'refusing to compare models with different vector sizes: '
            f'old has dim {old_kv.vector_size}, new has dim {new_kv.vector_size}'
        )
    old_index, old_spelling = _find_index(old_kv, key, normalizer)
    new_index, new_spelling = _find_index(new_kv, key, normalizer)
    nkey = normalizer.normalize(key)
    if old_index is None and new_index is None:
        raise KeyError(f"key {key!r} (normalized {nkey!r}) is absent from both versions")
    delta = KeyDelta(
        key=key, normalized_key=nkey, status='removed', old_spelling=old_spelling, new_spelling=new_spelling,
        vector_size=old_kv.vector_size,
    )
    if old_index is None:
        delta.status = 'added'
        if topn > 0:
            delta.new_nearest_neighbors = _neighbors(new_kv, new_kv.vectors[new_index], topn)
        return delta
    old_vector = old_kv.vectors[old_index]
    if new_index is None:
        delta.status = 'removed'
        if topn > 0:
            delta.old_nearest_neighbors = _neighbors(old_kv, old_vector, topn)
        return delta
    new_vector = new_kv.vectors[new_index]
    metrics = _single_delta(old_vector, new_vector, policy)
    delta.status = metrics['status']
    delta.l2_delta = metrics['l2_delta']
    delta.relative_l2_delta = metrics['relative_l2_delta']
    delta.cosine_similarity = metrics['cosine_similarity']
    delta.cosine_distance = metrics['cosine_distance']
    delta.old_norm = metrics['old_norm']
    delta.new_norm = metrics['new_norm']
    if topn > 0:
        delta.old_nearest_neighbors = _neighbors(old_kv, old_vector, topn, exclude_index=old_index)
        delta.new_nearest_neighbors = _neighbors(new_kv, new_vector, topn, exclude_index=new_index)
        old_top1 = delta.old_nearest_neighbors[0][0] if delta.old_nearest_neighbors else None
        new_top1 = delta.new_nearest_neighbors[0][0] if delta.new_nearest_neighbors else None
        delta.nearest_neighbor_changed = old_top1 != new_top1
        old_keys = [k for k, _ in delta.old_nearest_neighbors]
        new_keys = [k for k, _ in delta.new_nearest_neighbors]
        delta.nearest_neighbor_overlap = len(set(old_keys) & set(new_keys)) / max(1, len(set(old_keys) | set(new_keys)))
        if old_top1 in new_keys:
            delta.old_top1_rank_in_new = new_keys.index(old_top1) + 1
    return delta


# ---------------------------------------------------------------------------
# Similarity-task evaluation
# ---------------------------------------------------------------------------

def evaluate_similarity_tasks(kv: KeyedVectors, *, pairs=None, analogies=None,
                              pairs_kwargs=None, analogies_kwargs=None) -> Dict[str, Any]:
    """Evaluate one version on the standard gensim similarity tasks.

    Returns a JSON-serializable dict with ``pearson``, ``spearman``, ``oov_ratio``
    and optionally ``analogy_accuracy``.  This only wraps the existing
    :meth:`~gensim.models.keyedvectors.KeyedVectors.evaluate_word_pairs` and
    :meth:`~gensim.models.keyedvectors.KeyedVectors.evaluate_word_analogies`
    methods -- the evaluation definition is unchanged.
    """
    result = {}
    if pairs:
        pearson, spearman, oov = kv.evaluate_word_pairs(pairs, **(pairs_kwargs or {}))
        result['word_pairs'] = {
            'pearson': float(pearson[0]),
            'pearson_pvalue': float(pearson[1]),
            'spearman': float(spearman[0]),
            'spearman_pvalue': float(spearman[1]),
            'oov_ratio': float(oov),
            'file': os.fspath(pairs),
        }
    if analogies:
        score, _sections = kv.evaluate_word_analogies(analogies, **(analogies_kwargs or {}))
        result['analogies'] = {'accuracy': float(score), 'file': os.fspath(analogies)}
    return result


def compare_similarity_tasks(old, new, *, pairs=None, analogies=None,
                             pairs_kwargs=None, analogies_kwargs=None) -> Dict[str, Any]:
    """Run :func:`evaluate_similarity_tasks` on both versions and report deltas.

    Returns ``{'old': ..., 'new': ..., 'delta': ...}`` where ``delta`` contains
    ``new - old`` for every numeric metric.  Both models are loaded fully
    (mmap for paths); this is an evaluation step, not a streaming step.
    """
    old_kv = _load_kv(old, mmap=True)
    new_kv = _load_kv(new, mmap=True)
    old_result = evaluate_similarity_tasks(
        old_kv, pairs=pairs, analogies=analogies,
        pairs_kwargs=pairs_kwargs, analogies_kwargs=analogies_kwargs,
    )
    new_result = evaluate_similarity_tasks(
        new_kv, pairs=pairs, analogies=analogies,
        pairs_kwargs=pairs_kwargs, analogies_kwargs=analogies_kwargs,
    )
    delta = {}
    numeric = (int, float)
    comparables = {
        'word_pairs': ('pearson', 'spearman', 'oov_ratio'),
        'analogies': ('accuracy',),
    }
    for task in set(old_result) & set(new_result):
        metrics = comparables.get(task, tuple(old_result[task]))
        delta[task] = {}
        for metric in metrics:
            if metric not in old_result[task] or metric not in new_result[task]:
                continue
            old_value = old_result[task][metric]
            new_value = new_result[task][metric]
            if isinstance(old_value, numeric) and isinstance(new_value, numeric):
                delta[task][metric] = new_value - old_value
    return {'old': old_result, 'new': new_result, 'delta': delta}


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def _iter_updates(updates, normalizer, format='auto'):
    """Yield ``(key, vector)`` from a KeyedVectors, path, mapping or iterable."""
    if isinstance(updates, dict):
        for key, vector in updates.items():
            yield key, np.asarray(vector)
        return
    if isinstance(updates, KeyedVectors):
        for index, key in enumerate(updates.index_to_key):
            yield key, updates.vectors[index]
        return
    if hasattr(updates, 'wv') and isinstance(getattr(updates, 'wv'), KeyedVectors):
        yield from _iter_updates(updates.wv, normalizer, format)
        return
    if isinstance(updates, (str, bytes, os.PathLike)):
        for keys, vectors in iter_vectors(updates, format=format):
            for key, vector in zip(keys, vectors):
                yield key, vector
        return
    for key, vector in updates:
        yield key, np.asarray(vector)


def _expected_update_count(updates, format='auto'):
    """Best-effort count for pre-allocation; ``None`` when unknown."""
    if isinstance(updates, dict):
        return len(updates)
    if isinstance(updates, KeyedVectors):
        return len(updates.index_to_key)
    if hasattr(updates, 'wv') and isinstance(getattr(updates, 'wv'), KeyedVectors):
        return len(updates.wv.index_to_key)
    if isinstance(updates, (str, bytes, os.PathLike)):
        payload = os.fsdecode(os.fspath(updates))
        try:
            resolved = _detect_format(payload, format)
        except (ValueError, OSError):
            return None
        if resolved == 'kv':
            return len(_load_kv(payload, mmap=True))
        if resolved == 'word2vec-binary':
            # Counting binary records needs a full scan; skip it and grow the buffer.
            return None
        try:
            with utils.open(payload, 'rb') as fin:
                return max(0, int(fin.readline().split()[0]))
        except (ValueError, IndexError, UnicodeDecodeError):
            return None
    return None


def _check_supported_for_update(kv: KeyedVectors):
    if kv.__class__.__name__ == 'FastTextKeyedVectors':
        raise UnsupportedModelError(
            'incremental update of FastTextKeyedVectors is not supported: its public vectors are '
            'derived from vectors_vocab plus hashed ngram vectors, so replacing rows would desynchronize '
            'subword vectors. Rebuild the FastText model, or update a plain KeyedVectors.'
        )
    if kv.__class__.__name__ == 'PoincareKeyedVectors':
        raise UnsupportedModelError(
            'incremental update of PoincareKeyedVectors is not supported: downstream distance is '
            'hyperbolic and a plain L2/cosine delta is not a valid update criterion.'
        )


def _default_for_dtype(dtype):
    if dtype.kind == 'O':
        return None
    if dtype.kind in ('U', 'S'):
        return ''
    return 0


def _build_expandos(base_kv, surviving_indices, added_count, added_attrs):
    """Carry base per-key attributes over and extend them for appended keys."""
    total = len(surviving_indices) + added_count
    attributes = dict(base_kv.expandos)
    for attr in added_attrs:
        attributes.setdefault(attr, None)
    result = {}
    for attr, values in attributes.items():
        collected = added_attrs.get(attr)
        has_collected = collected is not None and len(collected) > 0
        if values is None:
            if not has_collected:
                continue
            array = np.asarray(collected)
            defaults = np.empty(total, dtype=array.dtype)
            defaults[:] = _default_for_dtype(array.dtype)
            defaults[len(surviving_indices):] = array
            result[attr] = defaults
            continue
        carried = values[surviving_indices]
        if added_count:
            extra = np.empty(added_count, dtype=carried.dtype)
            extra[:] = _default_for_dtype(carried.dtype)
            if has_collected:
                extra[:] = np.asarray(collected)
            carried = np.concatenate([carried, extra])
        result[attr] = carried
    return result


def _write_model_atomically(kv: KeyedVectors, output_path: str) -> List[str]:
    """Save ``kv`` under a temporary name and move into place; return written paths."""
    output_dir = os.path.dirname(os.path.abspath(output_path)) or '.'
    os.makedirs(output_dir, exist_ok=True)
    temp_base = os.path.join(output_dir, f'.{os.path.basename(output_path)}.tmp-{os.getpid()}')
    try:
        kv.save(temp_base)
        produced = [temp_base] + sorted(glob.glob(temp_base + '.*'))
        mapping = []
        for produced_file in produced:
            suffix = produced_file[len(temp_base):]
            mapping.append((produced_file, output_path + suffix))
        # Main pickle last, so a crash cannot leave a header pointing at missing arrays.
        mapping.sort(key=lambda pair: pair[1] == output_path)
        written = []
        for source, destination in mapping:
            os.replace(source, destination)
            written.append(destination)
        return written
    except Exception:
        for leftover in [temp_base] + sorted(glob.glob(temp_base + '.*')):
            try:
                if os.path.isfile(leftover):
                    os.unlink(leftover)
            except OSError:
                pass
        raise


def incremental_update(
        base, updates, output_path, *, normalizer=None, policy=None, format='auto',
        prune_missing=False, overwrite=False, backup_dir=None, backup=True,
        max_changed_keys=1_000,
    ) -> UpdateRecord:
    """Apply only changed/new vectors to a base model and write a new version.

    Unchanged base rows are reused **byte for byte** (no retraining, no
    recomputation); updated rows are replaced; new keys are appended.  Keys in
    the base that are absent from ``updates`` are kept unless ``prune_missing``
    is set -- a delta file normally contains only the changed part.

    Parameters
    ----------
    base : KeyedVectors or str or path-like
        The previous version.  Required: if missing,
        :class:`BaseVersionMissingError` is raised saying a full rebuild is the
        only option.
    updates : KeyedVectors, path, dict or iterable of (key, vector)
        The delta source: only vectors to replace/append.
    output_path : str
        Where the updated version is written.  May equal the base path when
        ``overwrite=True``.
    normalizer : KeyNormalizer, optional
        Key identity rules (default :class:`KeyNormalizer`).
    policy : ChangePolicy, optional
        Thresholds used when classifying a provided vector as bitwise identical.
    format : str, optional
        Format of ``base``/``updates`` paths when they are word2vec files.
    prune_missing : bool, optional
        Drop base keys that do not appear in ``updates`` (default False).
    overwrite : bool, optional
        Allow replacing an existing ``output_path`` (default False).
    backup : bool, optional
        Back up the previous output files before replacing them (default True).
        Ignored when the output does not yet exist.
    backup_dir : str, optional
        Directory for the previous version.  Defaults to ``output_path + '.prev'``.
    max_changed_keys : int, optional
        Maximum number of per-key change entries embedded in the update record.

    Returns
    -------
    UpdateRecord
        Counts, timing, output/backup paths and rollback information.  Use
        ``write_json`` to persist it.

    Raises
    ------
    BaseVersionMissingError
        If ``base`` does not exist.  Incremental update is impossible; rebuild.
    DimensionMismatchError
        If an update vector's size differs from the base vector size.
    UnsupportedModelError
        For FastText/Poincare vector classes.
    FileExistsError
        If ``output_path`` exists and ``overwrite`` is False.

    Notes
    -----
    Writing is atomic with respect to the final model: all files are produced
    under temporary names first and moved into place only after the save
    succeeded.  On any failure the temporary files are removed and, when the
    output existed, the previous version is restored from ``backup_dir``.
    Earlier partial replacements with no previous version are removed.

    The new model must be materialized in memory once (the output matrix).
    Load the base with ``mmap='r'`` (automatic for paths) so peak RAM is about
    one model rather than two.  The operation is single-threaded and
    deterministic; no randomness is used.  Concurrent writers to the same
    output path must be serialized by the caller.

    """
    started = time.time()
    normalizer = normalizer or KeyNormalizer()
    policy = policy or ChangePolicy()

    kind, payload = _as_source(base)
    if kind == 'path':
        _require_exists(payload)
    base_kv = _load_kv(base, mmap=True)
    _check_supported_for_update(base_kv)
    base_info, _base_limitations = _resolved_info(
        ('object', base_kv, payload if kind == 'path' else f'{base_kv.__class__.__name__} at 0x{id(base_kv):x}'),
    )
    base_info_dict = base_info.to_dict()
    base_info_dict['path'] = payload if kind == 'path' else None

    output_path = os.fspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"output {output_path!r} already exists; pass overwrite=True to replace it "
            '(the previous version is backed up first when backup=True)'
        )

    base_keys = list(base_kv.index_to_key)
    base_tokens = [_encode_identity(normalizer.normalize(key)) for key in base_keys]
    identity_to_index = {}
    collisions = 0
    for index, token in enumerate(base_tokens):
        if token in identity_to_index:
            collisions += 1
            continue
        identity_to_index[token] = index

    vector_size = base_kv.vector_size
    update_attr_sources = updates.expandos if isinstance(updates, KeyedVectors) else {}
    update_positions = updates.key_to_index if isinstance(updates, KeyedVectors) else {}

    # 1) Stream the supplied (key, vector) pairs.  Only *changes* are kept: bitwise
    #    identical vectors are dropped immediately, so memory is proportional to the
    #    delta size, not to the size of the delta source.
    provided_ids = set()
    changed_tokens: List[str] = []
    changed_row_keys: List[Any] = []
    added_keys: List[Any] = []
    added_attrs: Dict[str, List[Any]] = {}
    expected = _expected_update_count(updates, format)
    # Only changed rows are kept, so a full-size delta source must not preallocate a
    # full-size buffer.  Grow geometrically from a small start instead.
    capacity = min(max(expected or 1024, 1024), _INITIAL_DELTA_CAPACITY)
    changed_matrix = np.empty((capacity, vector_size), dtype=base_kv.vectors.dtype)
    changed_count = 0
    batch_base_indices: List[int] = []
    batch_vectors: List[np.ndarray] = []
    batch_tokens: List[str] = []
    batch_keys: List[Any] = []
    batch_size = 20_000
    changed_key_examples: List[Dict[str, Any]] = []

    def append_changed(vector, token, key, metrics=None):
        nonlocal changed_matrix, changed_count
        if changed_count == changed_matrix.shape[0]:
            grown = np.empty((changed_matrix.shape[0] * 2, vector_size), dtype=base_kv.vectors.dtype)
            grown[:changed_count] = changed_matrix[:changed_count]
            changed_matrix = grown
        changed_matrix[changed_count] = vector
        changed_tokens.append(token)
        changed_row_keys.append(key)
        changed_count += 1
        if metrics is not None and len(changed_key_examples) < max_changed_keys:
            relative = metrics['relative_l2_delta']
            cosine = metrics['cosine_distance']
            changed_key_examples.append({
                'key': key,
                'status': 'changed',
                'l2_delta': metrics['l2_delta'],
                'relative_l2_delta': None if math.isnan(relative) else relative,
                'cosine_distance': None if math.isnan(cosine) else cosine,
            })

    def flush_batch():
        if not batch_base_indices:
            return
        batch = _batch_delta(
            base_kv.vectors[np.asarray(batch_base_indices)], np.vstack(batch_vectors), policy,
        )
        for local in np.nonzero(batch['status'] != 0)[0]:  # 0 = bitwise identical -> reused
            append_changed(
                batch_vectors[local], batch_tokens[local], batch_keys[local],
                {
                    key: float(value[local])
                    for key, value in batch.items()
                    if key in ('l2_delta', 'relative_l2_delta', 'cosine_distance')
                },
            )
        batch_base_indices.clear()
        batch_vectors.clear()
        batch_tokens.clear()
        batch_keys.clear()

    for key, vector in _iter_updates(updates, normalizer, format=format):
        vector = np.asarray(vector)
        if vector.ndim != 1 or vector.shape[0] != vector_size:
            raise DimensionMismatchError(
                f"update vector for key {key!r} has shape {vector.shape}, expected ({vector_size},); "
                'refusing to apply an update with a different vector size'
            )
        token = _encode_identity(normalizer.normalize(key))
        if token in provided_ids:
            collisions += 1
            continue
        provided_ids.add(token)
        index = identity_to_index.get(token)
        if index is None:
            append_changed(vector, token, key)
            added_keys.append(key)
            if update_attr_sources:
                update_index = update_positions[key]
                for attr, values in update_attr_sources.items():
                    added_attrs.setdefault(attr, []).append(values[update_index])
            continue
        batch_base_indices.append(index)
        batch_vectors.append(np.array(vector, copy=True))
        batch_tokens.append(token)
        batch_keys.append(key)
        if len(batch_base_indices) >= batch_size:
            flush_batch()
    flush_batch()
    changed_matrix = changed_matrix[:changed_count]
    rows_added = len(added_keys)

    if prune_missing:
        surviving_indices = np.asarray(
            [i for i, token in enumerate(base_tokens) if token in provided_ids], dtype=np.int64,
        )
        rows_pruned = len(base_keys) - len(surviving_indices)
    else:
        surviving_indices = np.arange(len(base_keys), dtype=np.int64)
        rows_pruned = 0

    # 2) Assemble the result matrix in bulk: carry base rows, overwrite changed rows,
    #    append new rows.  No per-row Python loop over the unchanged tail.
    if update_attr_sources:
        for attr, values in update_attr_sources.items():
            if added_keys:
                source_indices = [update_positions[key] for key in added_keys]
                added_attrs[attr] = values[np.asarray(source_indices)]

    total_rows = len(surviving_indices) + rows_added
    rows_replaced = 0
    added_offset = 0
    surviving_position = None
    if prune_missing and changed_count:
        surviving_position = {int(index): position for position, index in enumerate(surviving_indices)}
    if not prune_missing and not rows_added and not changed_count:
        # Pure reuse: the output is the base matrix unchanged.  Zero-copy for an
        # in-memory caller (save copies to disk); the loaded mmap array is copied.
        result_matrix = base_kv.vectors if isinstance(base, KeyedVectors) else np.asarray(base_kv.vectors)
    else:
        result_matrix = np.empty((total_rows, vector_size), dtype=base_kv.vectors.dtype)
        if prune_missing:
            result_matrix[: len(surviving_indices)] = base_kv.vectors[surviving_indices]
        else:
            # Same order: bulk-copy straight out of the (possibly mmap'd) base.
            result_matrix[: len(surviving_indices)] = base_kv.vectors
        for position, token in enumerate(changed_tokens):
            index = identity_to_index.get(token)
            if index is None:
                result_matrix[len(surviving_indices) + added_offset] = changed_matrix[position]
                added_offset += 1
                continue
            destination = int(index) if surviving_position is None else surviving_position.get(int(index))
            if destination is None:
                continue  # changed row pruned from the output
            result_matrix[destination] = changed_matrix[position]
            rows_replaced += 1
    del changed_matrix

    result_keys = [base_keys[index] for index in surviving_indices] + added_keys
    # Every matched token maps to exactly one base index (the first occurrence in the
    # base wins under normalization), so these counts are exact even with base collisions.
    rows_reused_exact = sum(1 for token in provided_ids if token in identity_to_index) - rows_replaced
    rows_unchanged_kept = len(surviving_indices) - rows_replaced - rows_reused_exact

    result = KeyedVectors(vector_size, count=len(result_keys), dtype=base_kv.vectors.dtype)
    result.index_to_key = result_keys
    result.next_index = len(result_keys)
    result.key_to_index = {key: i for i, key in enumerate(result_keys)}
    result.vectors = result_matrix
    result.norms = None
    result.expandos = _build_expandos(base_kv, surviving_indices, rows_added, added_attrs)

    # Atomically publish the new version, keeping the previous one as a backup.
    previous_exists = os.path.exists(output_path)
    actual_backup = None
    written = []
    try:
        if previous_exists and backup:
            actual_backup = backup_dir or (output_path + '.prev')
            if os.path.exists(actual_backup):
                raise FileExistsError(
                    f'backup directory {actual_backup!r} already exists; move it away or pass a different '
                    'backup_dir'
                )
            os.makedirs(actual_backup)
            for existing in [output_path] + sorted(glob.glob(output_path + '.*')):
                if os.path.isfile(existing):
                    _link_or_copy(existing, os.path.join(actual_backup, os.path.basename(existing)))
        written = _write_model_atomically(result, output_path)
    except Exception:
        if written and not actual_backup:
            for path in written:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        if actual_backup and os.path.isdir(actual_backup):
            try:
                restore_backup(output_path, actual_backup, remove_backup=False)
            except Exception:  # pragma: no cover - defensive
                logger.exception("failed to restore %s from %s", output_path, actual_backup)
        raise

    elapsed = time.time() - started
    if isinstance(updates, (str, bytes, os.PathLike)):
        updates_label = os.fspath(updates)
    elif isinstance(updates, KeyedVectors):
        updates_label = f'{updates.__class__.__name__} at 0x{id(updates):x}'
    elif isinstance(updates, dict):
        updates_label = 'dict'
    else:
        updates_label = type(updates).__name__
    return UpdateRecord(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base=base_info_dict,
        updates={'label': updates_label, 'provided': len(provided_ids)},
        output_path=os.path.abspath(output_path),
        backup_dir=os.path.abspath(actual_backup) if actual_backup else None,
        normalizer=normalizer.describe(),
        policy=policy.describe(),
        rows_total=len(result_keys),
        rows_reused_exact=rows_reused_exact,
        rows_replaced=rows_replaced,
        rows_added=rows_added,
        rows_unchanged_kept=rows_unchanged_kept,
        rows_pruned=rows_pruned,
        normalization_collisions=collisions,
        changed_keys=changed_key_examples,
        elapsed_seconds=elapsed,
        limitations=[
            'the updated model is a KeyedVectors (Word2Vec/FastText/Doc2Vec training state is not carried over).',
            'FastText ngram vectors and Poincare hyperbolic distances are not supported for updates.',
            'concurrent writers to the same output path are not coordinated; use external locking.',
            'the output matrix is materialized in memory once; load the base with mmap to avoid two full copies.',
        ],
    )


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def restore_backup(output_path, backup_dir, *, remove_backup=False) -> List[str]:
    """Restore a previous version kept by :func:`incremental_update`.

    Copies every file from ``backup_dir`` back next to ``output_path`` and
    returns the restored paths.  Set ``remove_backup=True`` to delete the
    backup directory afterwards.
    """
    output_path = os.fspath(output_path)
    backup_dir = os.fspath(backup_dir)
    if not os.path.isdir(backup_dir):
        raise FileNotFoundError(f'backup directory {backup_dir!r} does not exist')
    restored = []
    for source in sorted(glob.glob(os.path.join(backup_dir, '*'))):
        destination = os.path.join(os.path.dirname(os.path.abspath(output_path)), os.path.basename(source))
        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            if os.path.exists(destination) and os.path.samefile(source, destination):
                # Backup was made with a hard link and the output was never replaced,
                # so it already holds the backed-up content.
                restored.append(destination)
                continue
            shutil.copy2(source, destination)
        restored.append(destination)
    if remove_backup:
        shutil.rmtree(backup_dir)
    logger.info("restored %d files from %s", len(restored), backup_dir)
    return restored
