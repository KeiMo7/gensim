.. _vector_delta:

========================================================
Comparing word-vector versions and incremental updating
========================================================

Motivation
----------

After a word-vector model is retrained, two things can change at once:

* the **vocabulary** (tokens are added or removed), and
* the **vectors** of the tokens that survive.

Downstream similarity results can move because of either, and a full rebuild
makes the two impossible to tell apart.  :mod:`gensim.models.vector_delta` adds
an opt-in layer on top of the existing :class:`~gensim.models.keyedvectors.KeyedVectors`
storage that:

* compares two versions and reports *what* changed (vocabulary delta, vector
  delta beyond a threshold, per-key offsets and nearest-neighbour changes),
* evaluates both versions on the standard similarity tasks,
* applies only the changed part of a new model, reusing unchanged rows byte for
  byte, and
* keeps the previous version so a failed update can be rolled back.

Nothing in the existing training, saving or loading interfaces changes.  If you
do not call the new functions, behaviour is identical to previous releases.

Quick start
-----------

.. sourcecode:: pycon

    >>> from gensim.models.vector_delta import compare_versions, explain_key, incremental_update
    >>> report = compare_versions('wiki-2025.kv', 'wiki-2026.kv', streaming=True)
    >>> report.write_json('wiki-diff.json')
    >>> report.write_markdown('wiki-diff.md')

    >>> delta = explain_key('wiki-2025.kv', 'wiki-2026.kv', 'New York', topn=10)
    >>> delta.status, delta.relative_l2_delta, delta.nearest_neighbor_changed

    >>> record = incremental_update(
    ...     'wiki-2025.kv', 'wiki-2026.kv', 'wiki-2026.incremental.kv',
    ...     backup_dir='wiki-2025.backup', overwrite=True,
    ... )
    >>> record.write_json('wiki-update.json')

The same operations are available from the command line::

    python -m gensim.scripts.vector_delta compare old.kv new.kv \
        --report-json diff.json --report-markdown diff.md
    python -m gensim.scripts.vector_delta explain old.kv new.kv --key "New York"
    python -m gensim.scripts.vector_delta update old.kv new.kv updated.kv \
        --backup-dir updated.kv.prev --record update.json --overwrite
    python -m gensim.scripts.vector_delta restore updated.kv --backup-dir updated.kv.prev

Key identity
------------

Keys are matched after :class:`~gensim.models.vector_delta.KeyNormalizer` is
applied.  The rules, in order:

1. Unicode normalization (``NFC`` by default),
2. leading/trailing whitespace stripped,
3. internal whitespace runs collapsed to one space,
4. optional Unicode case folding (``case_sensitive=False`` uses ``str.casefold()``,
   not ``lower()``).

Non-string keys (integer Doc2Vec tags) are left unchanged.  When two keys of the
same version normalize to the same identity, **the first occurrence in source
order wins** -- the rule the word2vec loaders already use for duplicate words --
and the collision count is reported (``old_normalization_collisions`` /
``new_normalization_collisions``), never silently dropped.

Metrics and thresholds
----------------------

Vectors are first checked for **bitwise equality**.  If the bytes differ, three
metrics are reported per key:

====================== ======================================================
``l2_delta``           ``||v_new - v_old||`` (Euclidean)
``relative_l2_delta``  ``l2_delta / ||v_old||`` (scale-aware)
``cosine_distance``    ``1 - cos(v_old, v_new)`` (direction-aware)
====================== ======================================================

A key counts as **changed** when::

    not bitwise-equal AND (relative_l2_delta > relative_l2_threshold
                           OR cosine_distance > cosine_distance_threshold)

The defaults (``relative_l2_threshold=1e-3``, ``cosine_distance_threshold=1e-4``)
are four orders of magnitude above float32 representation noise (relative
epsilon about ``1.2e-7``) and far below the movement produced by a normal
retrain, so last-bit rounding differences fall into
``unchanged_within_threshold`` rather than being reported as real changes.  Tune
:class:`~gensim.models.vector_delta.ChangePolicy` if your model behaves
differently.  If an old vector has zero length, ``relative_l2_delta`` is
reported as ``null`` and any non-zero difference counts as changed.

Streaming and memory
--------------------

``streaming=True`` never holds both matrices in RAM.  The old version is spilled
to a temporary SQLite file on disk, the new version is streamed through it in
chunks, and only fixed-size statistics (including a deterministic reservoir
sample used for percentiles) are kept in memory.  Gensim ``.kv`` inputs are
opened with ``mmap='r'``; word2vec text/binary inputs are read with bounded
buffers.  Peak RAM is therefore governed by the chunk size, not the vocabulary
size.  The temporary database is deleted when the comparison finishes.

``streaming=None`` (the default) enables streaming automatically when either
source is a path; two in-memory ``KeyedVectors`` objects are compared in memory.

Similarity-task evaluation
--------------------------

:func:`~gensim.models.vector_delta.evaluate_similarity_tasks` and
:func:`~gensim.models.vector_delta.compare_similarity_tasks` wrap the existing
``evaluate_word_pairs`` / ``evaluate_word_analogies`` methods, so the task
definition is unchanged; they only add the before/after/delta structure::

    report = compare_versions(
        'old.kv', 'new.kv',
        similarity_tasks={'pairs': 'wordsim353.tsv', 'analogies': 'questions-words.txt'},
    )

Incremental update
------------------

:func:`~gensim.models.vector_delta.incremental_update` takes a base model and a
*delta* source (only changed/new vectors) and produces a new ``KeyedVectors``:

* unchanged base rows are copied **byte for byte** (no retraining, no
  recomputation, no rounding),
* vectors provided for existing keys replace those rows,
* new keys are appended,
* base keys absent from the delta are kept unless ``prune_missing=True``.

The output is written under temporary names and moved into place only after the
save succeeds.  If the output already exists, the previous version is hard-linked
or copied into ``backup_dir`` (default ``<output>.prev``) first; a failure during
the update restores that backup and leaves the original model unchanged.

Rollback is available programmatically through
:func:`~gensim.models.vector_delta.restore_backup` and on the command line via
``restore``.

Parameters
~~~~~~~~~~

``base``
    Previous version: ``KeyedVectors``, a full model exposing ``.wv``, or a path.
``updates``
    Delta source: ``KeyedVectors``/full model, path, ``dict`` or iterable of
    ``(key, vector)``.
``output_path``
    Where the updated model is written.
``normalizer`` / ``policy``
    Key identity and change thresholds (see above).
``prune_missing``
    Drop base keys not present in the delta (default ``False``).
``overwrite``
    Allow replacing an existing output (default ``False``).
``backup`` / ``backup_dir``
    Keep the previous output for rollback (default on; directory defaults to
    ``<output>.prev``).
``max_changed_keys``
    Cap on per-key entries embedded in the update record (default 1000).

Known limitations
-----------------

* The updated model is a plain ``KeyedVectors``.  ``Word2Vec``/``Doc2Vec``
  training state (syn1neg, vocabulary counts used for sampling, etc.) is **not**
  carried over; if you need to continue training, use the original model.
* ``FastTextKeyedVectors`` is not supported for update: its public vectors are
  derived from ``vectors_vocab`` plus hashed ngram vectors, so replacing rows
  would desynchronize the subword vectors.  Comparison uses the derived
  full-word vectors only.
* ``PoincareKeyedVectors`` is not supported for update, and its comparison
  deltas are advisory only (downstream distance is hyperbolic).
* ``compare_versions`` with word2vec-format input compares vector data only;
  vocabulary counts and other per-key attributes are not persisted in that
  format.
* ``explain_key`` needs both models loaded (mmap for paths) and computes
  float64 scratch for the full vocabulary, so it is meant for interactive
  queries, not million-scale per-key loops.
* Incremental update materializes one new output matrix in memory.  Load the
  base from a path so it is mmapped and peak RAM is about one model rather than
  two.
* Concurrent writers to the same output path are not coordinated; serialize
  them with external locking.

Concurrency and determinism
---------------------------

Nothing here introduces threads or processes, and nothing uses unseeded
randomness.  Comparisons are single-threaded and deterministic: source order
decides duplicate-identity winners and the percentile reservoir uses a fixed
seed.  Incremental updates are single-threaded and fully deterministic.  The
only concurrency caveat is external: do not modify source files while a
comparison is reading them, and do not run two updates against one output path
simultaneously.

Compatibility boundaries
------------------------

* Existing ``KeyedVectors`` files (including pre-4.0 models up-converted on
  load), word2vec text/binary files, and full models saved by this gensim
  version all work as inputs.
* Files produced by the incremental update are ordinary ``KeyedVectors`` saves
  and load with the existing ``KeyedVectors.load``; no adapter is needed.
* Old pickled models that this gensim version cannot load are not made loadable
  by this feature -- load them with a compatible gensim version and re-save
  first.
* The comparison report and update record JSON schemas are versioned
  (``schema_version``); they are new artifacts, not a modification of any
  existing format.

Reference
---------

The full API lives in :mod:`gensim.models.vector_delta` (see the API reference),
and the command line front-end in :mod:`gensim.scripts.vector_delta`.

Main entry points:

* :func:`gensim.models.vector_delta.compare_versions` -- two-version report
* :func:`gensim.models.vector_delta.explain_key` -- per-key offset and neighbour change
* :func:`gensim.models.vector_delta.incremental_update` -- delta-only update
* :func:`gensim.models.vector_delta.restore_backup` -- rollback
* :func:`gensim.models.vector_delta.evaluate_similarity_tasks` /
  :func:`gensim.models.vector_delta.compare_similarity_tasks` -- before/after metrics
* :func:`gensim.models.vector_delta.iter_vectors` -- chunked streaming iteration
* :class:`gensim.models.vector_delta.KeyNormalizer` -- key identity rules
* :class:`gensim.models.vector_delta.ChangePolicy` -- metrics and thresholds
* :class:`gensim.models.vector_delta.VersionComparisonReport`,
  :class:`gensim.models.vector_delta.KeyDelta`,
  :class:`gensim.models.vector_delta.UpdateRecord` -- report objects
