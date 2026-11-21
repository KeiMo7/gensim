#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jan Zikes, Radim Rehurek
# Copyright (C) 2014 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Online Latent Dirichlet Allocation (LDA) in Python, using all CPU cores to parallelize and speed up model training.

The parallelization uses multiprocessing; in case this doesn't work for you for some reason,
try the :class:`gensim.models.ldamodel.LdaModel` class which is an equivalent, but more straightforward and single-core
implementation.

The training algorithm:

* is **streamed**: training documents may come in sequentially, no random access required,
* runs in **constant memory** w.r.t. the number of documents: size of the
  training corpus does not affect memory footprint, can process corpora larger than RAM

Wall-clock `performance on the English Wikipedia <https://radimrehurek.com/gensim/wiki.html>`_ (2G corpus positions,
3.5M documents, 100K features, 0.54G non-zero entries in the final bag-of-words matrix), requesting 100 topics:


====================================================== ==============
 algorithm                                             training time
====================================================== ==============
 LdaMulticore(workers=1)                               2h30m
 LdaMulticore(workers=2)                               1h24m
 LdaMulticore(workers=3)                               1h6m
 old LdaModel()                                        3h44m
 simply iterating over input corpus = I/O overhead     20m
====================================================== ==============

(Measured on `this i7 server <http://www.hetzner.de/en/hosting/produkte_rootserver/ex40ssd>`_
with 4 physical cores, so that optimal `workers=3`, one less than the number of cores.)

This module allows both LDA model estimation from a training corpus and inference of topic distribution on new,
unseen documents. The model can also be updated with new documents for online training.

The core estimation code is based on the `onlineldavb.py script
<https://github.com/blei-lab/onlineldavb/blob/master/onlineldavb.py>`_, by
Matthew D. Hoffman, David M. Blei, Francis Bach:
`'Online Learning for Latent Dirichlet Allocation', NIPS 2010`_.

.. _'Online Learning for Latent Dirichlet Allocation', NIPS 2010: online-lda_
.. _'Online Learning for LDA' by Hoffman et al.: online-lda_
.. _online-lda: https://papers.neurips.cc/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf

Usage examples
--------------
The constructor estimates Latent Dirichlet Allocation model parameters based on a training corpus

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_corpus, common_dictionary
    >>>
    >>> lda = LdaMulticore(common_corpus, id2word=common_dictionary, num_topics=10)

Save a model to disk, or reload a pre-trained model

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> # Save model to disk.
    >>> temp_file = datapath("model")
    >>> lda.save(temp_file)
    >>>
    >>> # Load a potentially pretrained model from disk.
    >>> lda = LdaModel.load(temp_file)

Query, or update the model using new, unseen documents

.. sourcecode:: pycon

    >>> other_texts = [
    ...     ['computer', 'time', 'graph'],
    ...     ['survey', 'response', 'eps'],
    ...     ['human', 'system', 'computer']
    ... ]
    >>> other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
    >>>
    >>> unseen_doc = other_corpus[0]
    >>> vector = lda[unseen_doc]  # get topic probability distribution for a document
    >>>
    >>> # Update the model by incrementally training on the new corpus.
    >>> lda.update(other_corpus)  # update the LDA model with additional documents

Reproducible training, checkpointing and resuming
-------------------------------------------------
By default, :class:`LdaMulticore` behaves exactly as it always has: results may
vary between runs even with a fixed ``random_state``, because the E-step chunks
are assigned to worker processes by OS scheduling and the sufficient statistics
are merged in arrival order.

Passing ``deterministic=True`` (which requires an explicit integer
``random_state``) makes training fully reproducible for a fixed
``(random_state, workers, chunksize)`` configuration:

* each chunk's randomness is derived from ``(seed, pass, chunk_no)`` via
  :class:`numpy.random.SeedSequence`, so it does not depend on which worker
  process happens to receive the chunk,
* chunk results are merged strictly in chunk order, so the M-step schedule and
  all floating-point summation orders are fixed.

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_corpus, common_dictionary
    >>>
    >>> lda = LdaMulticore(
    ...     common_corpus, id2word=common_dictionary, num_topics=10,
    ...     workers=3, random_state=42, deterministic=True,
    ...     checkpoint_path='/tmp/lda.ckpt',  # optional, atomic per-pass checkpoints
    ... )

To resume an interrupted training run (bit-identical to training in one go,
when ``deterministic=True``):

.. sourcecode:: pycon

    >>> lda = LdaMulticore.resume_from_checkpoint('/tmp/lda.ckpt', common_corpus)

To quantify how much the result changes when the number of workers changes:

.. sourcecode:: pycon

    >>> from gensim.models.ldamulticore import compare_topic_word_distributions
    >>> metrics = compare_topic_word_distributions(lda_workers1, lda_workers3)
    >>> metrics['mean_symmetric_kl']  # doctest: +SKIP

Known limitations
-----------------
* Bitwise reproducibility is guaranteed for a fixed ``(random_state, workers,
  chunksize)`` configuration on the same machine (same numpy/BLAS build). Changing
  `workers` or `chunksize` changes the update schedule and therefore the result;
  use :func:`compare_topic_word_distributions` to quantify the difference.
* Deterministic mode uses synchronous rounds: each round of ``workers`` chunks is
  computed, collected and merged before the next round starts. This removes the
  pipelining overlap of the legacy mode and is typically ~30-50% slower; memory
  usage is unchanged (at most one round of intermediate results is buffered, the
  same order as the legacy result queue).
* ``deterministic=True`` requires an explicit integer ``random_state``; the seed is
  stored in ``model.training_metadata`` and travels with the saved model.
* Resuming requires a re-iterable corpus (it is hashed once for verification, then
  iterated for the remaining passes). The corpus digest is computed from the
  vocabulary and the word frequencies, never from file names or timestamps.
* Models and checkpoints are not interchangeable across these modes: a checkpoint
  written without ``deterministic=True`` can be resumed, but the result is not
  guaranteed to be bit-identical to an uninterrupted run, and the corpus check is
  only available if the checkpoint contains a corpus digest. Checkpoints written by
  this feature cannot be read by older gensim versions; regular ``save()`` files
  remain fully compatible both ways (older models simply lack the new metadata
  attributes and train with the legacy, non-deterministic behavior).
* In deterministic mode, repeated ``update()`` calls form one continuous training
  trajectory over the *same* corpus (pass numbering and the document count continue
  across calls). To train on new data, start a new model.

"""

import hashlib
import logging
import os
import pickle
import queue
import struct
from multiprocessing import Pool, Queue, cpu_count

import numpy as np

from gensim import utils
from gensim.models.ldamodel import LdaModel, LdaState


logger = logging.getLogger(__name__)

#: Magic header of checkpoint files written by :meth:`LdaMulticore._save_checkpoint`.
_CHECKPOINT_MAGIC = b'gensim-ldamulticore-checkpoint-v1\n'

#: Format tag stored inside the checkpoint payload.
_CHECKPOINT_FORMAT = 'ldamulticore-checkpoint-v1'


class CheckpointError(Exception):
    """Base class for checkpoint related errors."""


class CheckpointCorruptError(CheckpointError, IOError):
    """Raised when a checkpoint file cannot be read, is truncated or fails its integrity check.

    The error message names the part of the file that is broken (header, checksum or payload)
    and whether a previous generation of the checkpoint is available for partial recovery.
    """


class CorpusMismatchError(CheckpointError, ValueError):
    """Raised when the corpus supplied to :meth:`LdaMulticore.resume_from_checkpoint` does not
    match the corpus digest stored in the checkpoint. Resuming with a different corpus would
    mix incomparable data into the model, so it is refused."""


def _vocab_digest_bytes(id2word):
    """Hash the vocabulary: the token<->id mapping and, for a :class:`~gensim.corpora.dictionary.Dictionary`,
    the document frequencies. Content-based; file metadata (timestamps, paths) is never used.

    """
    hasher = hashlib.sha256()
    if id2word is None:
        hasher.update(b'<no-vocab>\0')
        return hasher.digest()
    token2id = getattr(id2word, 'token2id', None)
    if token2id is not None:
        pairs = sorted((int(idx), str(token)) for token, idx in token2id.items())
    else:
        pairs = sorted((int(idx), str(token)) for idx, token in id2word.items())
    for idx, token in pairs:
        hasher.update(struct.pack('<q', idx))
        hasher.update(token.encode('utf-8'))
        hasher.update(b'\0')
    dfs = getattr(id2word, 'dfs', None)
    if dfs:
        for idx, df in sorted(dfs.items()):
            hasher.update(struct.pack('<q', int(idx)))
            hasher.update(struct.pack('<q', int(df)))
    return hasher.digest()


def _new_corpus_hasher(id2word):
    """Start a corpus digest hasher, seeded with the vocabulary digest."""
    hasher = hashlib.sha256()
    hasher.update(b'gensim-lda-corpus-digest-v1\0')
    hasher.update(_vocab_digest_bytes(id2word))
    return hasher


def _update_corpus_hasher(hasher, corpus):
    """Feed a chunk (or whole corpus) of bag-of-words documents into the digest hasher.

    Hashes each document's length and its (term_id, count) pairs in corpus order, so the
    digest captures the vocabulary and the word frequencies of the actual training data.

    """
    for doc in corpus:
        hasher.update(struct.pack('<i', len(doc)))
        for term_id, count in doc:
            hasher.update(struct.pack('<q', int(term_id)))
            hasher.update(struct.pack('<d', float(count)))


def corpus_digest(corpus, id2word=None):
    """Compute a content-based digest of a training corpus.

    The digest covers the vocabulary (token<->id mapping and document frequencies, if
    `id2word` is a :class:`~gensim.corpora.dictionary.Dictionary`) and every document's
    (term_id, count) pairs in corpus order. It deliberately does **not** depend on file
    paths or modification times, so two corpora with identical content always produce the
    same digest regardless of where they were read from.

    Parameters
    ----------
    corpus : iterable of list of (int, float)
        The corpus to hash. Must be re-iterable if you need it afterwards (it is consumed once).
    id2word : {dict of (int, str), :class:`gensim.corpora.dictionary.Dictionary`}, optional
        Vocabulary to include in the digest.

    Returns
    -------
    str
        Hex digest string.

    """
    hasher = _new_corpus_hasher(id2word)
    _update_corpus_hasher(hasher, corpus)
    return hasher.hexdigest()


def _chunk_seed(seed, pass_no, chunk_no):
    """Derive the random seed for one E-step chunk from (seed, pass, chunk number).

    Deterministic across runs, processes and platforms, and independent of which worker
    process ends up executing the chunk.

    """
    return np.random.SeedSequence([int(seed), int(pass_no), int(chunk_no)]).generate_state(1)[0]


def compare_topic_word_distributions(model_a, model_b, topn=20):
    """Quantify the difference between the topic-word distributions of two trained LDA models.

    Topics are aligned between the models with the Hungarian method (minimum-cost matching on
    cosine distance), because identical training data can still permute topic indices when
    hyper-parameters such as `workers` change.

    Parameters
    ----------
    model_a, model_b : :class:`~gensim.models.ldamodel.LdaModel`
        The two trained models to compare. Must have the same number of topics.
    topn : int, optional
        Number of top words per topic used for the word-overlap metric.

    Returns
    -------
    dict
        Metrics dictionary with keys:

        * ``mean_cosine_similarity`` / ``min_cosine_similarity``: cosine similarity between
          matched topic-word distributions (1.0 = identical),
        * ``mean_symmetric_kl`` / ``max_symmetric_kl``: symmetric KL divergence (nats) between
          matched topics (0.0 = identical),
        * ``mean_top_word_overlap``: mean Jaccard overlap of the `topn` top words per matched
          topic pair (1.0 = identical),
        * ``topic_mapping``: list of ``(topic_a, topic_b)`` index pairs.

    """
    from scipy.optimize import linear_sum_assignment

    topics_a = model_a.get_topics().astype(np.float64)
    topics_b = model_b.get_topics().astype(np.float64)
    if topics_a.shape != topics_b.shape:
        raise ValueError(
            "models are not comparable: topic-word shapes differ, %s vs %s" %
            (topics_a.shape, topics_b.shape)
        )
    topics_a /= topics_a.sum(axis=1, keepdims=True)
    topics_b /= topics_b.sum(axis=1, keepdims=True)

    norms = np.linalg.norm(topics_a, axis=1)[:, None] * np.linalg.norm(topics_b, axis=1)[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        similarity = np.nan_to_num(topics_a @ topics_b.T / norms)
    row_ind, col_ind = linear_sum_assignment(1.0 - similarity)

    eps = 1e-12
    sym_kl = 0.5 * (
        np.sum(topics_a[row_ind] * np.log((topics_a[row_ind] + eps) / (topics_b[col_ind] + eps)), axis=1)
        + np.sum(topics_b[col_ind] * np.log((topics_b[col_ind] + eps) / (topics_a[row_ind] + eps)), axis=1)
    )

    overlaps = []
    for a_idx, b_idx in zip(row_ind, col_ind):
        top_a = set(np.argsort(topics_a[a_idx])[::-1][:topn])
        top_b = set(np.argsort(topics_b[b_idx])[::-1][:topn])
        union = top_a | top_b
        overlaps.append(len(top_a & top_b) / len(union) if union else 1.0)

    matched_similarity = similarity[row_ind, col_ind]
    return {
        'mean_cosine_similarity': float(np.mean(matched_similarity)),
        'min_cosine_similarity': float(np.min(matched_similarity)),
        'mean_symmetric_kl': float(np.mean(sym_kl)),
        'max_symmetric_kl': float(np.max(sym_kl)),
        'mean_top_word_overlap': float(np.mean(overlaps)),
        'topic_mapping': list(zip(row_ind.tolist(), col_ind.tolist())),
    }


def compare_training_metadata(model_a, model_b):
    """Compare the training metadata of two models, to judge whether their results are comparable.

    Returns
    -------
    dict
        Mapping of metadata field name to ``(value_a, value_b, matches)`` triples. Two models
        trained on the same corpus with the same configuration have ``matches=True`` for the
        ``corpus_digest``, ``num_topics``, ``chunksize``, ``passes`` and ``random_seed`` fields.

    """
    meta_a = getattr(model_a, 'training_metadata', None) or {}
    meta_b = getattr(model_b, 'training_metadata', None) or {}
    keys = sorted((set(meta_a) | set(meta_b)) - {'per_pass_perplexity'})
    return {
        key: (meta_a.get(key), meta_b.get(key), meta_a.get(key) == meta_b.get(key))
        for key in keys
    }


class LdaMulticore(LdaModel):
    """An optimized implementation of the LDA algorithm, able to harness the power of multicore CPUs.
    Follows the similar API as the parent class :class:`~gensim.models.ldamodel.LdaModel`.

    """
    # Class-level defaults, so that models pickled before these attributes existed
    # can still be loaded and used with this version of the code.
    deterministic = False
    checkpoint_path = None
    random_seed = None
    training_metadata = None
    _passes_completed = 0
    _corpus_digest = None
    _corpus_hasher = None

    def __init__(self, corpus=None, num_topics=100, id2word=None, workers=None,
                 chunksize=2000, passes=1, batch=False, alpha='symmetric',
                 eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50,
                 gamma_threshold=0.001, random_state=None, minimum_probability=0.01,
                 minimum_phi_value=0.01, per_word_topics=False, dtype=np.float32,
                 deterministic=False, checkpoint_path=None):
        """

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
            If not given, the model is left untrained (presumably because you want to call
            :meth:`~gensim.models.ldamodel.LdaModel.update` manually).
        num_topics : int, optional
            The number of requested latent topics to be extracted from the training corpus.
        id2word : {dict of (int, str),  :class:`gensim.corpora.dictionary.Dictionary`}
            Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for
            debugging and topic printing.
        workers : int, optional
            Number of workers processes to be used for parallelization. If None all available cores
            (as estimated by `workers=cpu_count()-1` will be used. **Note** however that for
            hyper-threaded CPUs, this estimation returns a too high number -- set `workers`
            directly to the number of your **real** cores (not hyperthreads) minus one, for optimal performance.
            Must be a positive integer; invalid values are rejected immediately, at construction time.
        chunksize :  int, optional
            Number of documents to be used in each training chunk.
        passes : int, optional
            Number of passes through the corpus during training.
        alpha : {float, numpy.ndarray of float, list of float, str}, optional
            A-priori belief on document-topic distribution, this can be:
                * scalar for a symmetric prior over document-topic distribution,
                * 1D array of length equal to num_topics to denote an asymmetric user defined prior for each topic.

            Alternatively default prior selecting strategies can be employed by supplying a string:
                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,
                * 'asymmetric': Uses a fixed normalized asymmetric prior of `1.0 / (topic_index + sqrt(num_topics))`.
        eta : {float, numpy.ndarray of float, list of float, str}, optional
            A-priori belief on topic-word distribution, this can be:
                * scalar for a symmetric prior over topic-word distribution,
                * 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,
                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.

            Alternatively default prior selecting strategies can be employed by supplying a string:
                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,
                * 'auto': Learns an asymmetric prior from the corpus.
        decay : float, optional
            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten
            when each new document is examined. Corresponds to :math:`\\kappa` from
            `'Online Learning for LDA' by Hoffman et al.`_
        offset : float, optional
            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.
            Corresponds to :math:`\\tau_0` from `'Online Learning for LDA' by Hoffman et al.`_
        eval_every : int, optional
            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.
        iterations : int, optional
            Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.
        gamma_threshold : float, optional
            Minimum change in the value of the gamma parameters to continue iterating.
        minimum_probability : float, optional
            Topics with a probability lower than this threshold will be filtered out.
        random_state : {np.random.RandomState, int}, optional
            Either a randomState object or a seed to generate one. Useful for reproducibility.
            Note that results can still vary due to non-determinism in OS scheduling of the worker processes,
            unless `deterministic=True` is used (in which case `random_state` must be an explicit integer seed).
        minimum_phi_value : float, optional
            if `per_word_topics` is True, this represents a lower bound on the term probabilities.
        per_word_topics : bool
            If True, the model also computes a list of topics, sorted in descending order of most likely topics for
            each word, along with their phi values multiplied by the feature length (i.e. word count).
        dtype : {numpy.float16, numpy.float32, numpy.float64}, optional
            Data-type to use during calculations inside model. All inputs are also converted.
        deterministic : bool, optional
            If True, training is fully reproducible for a fixed (`random_state`, `workers`, `chunksize`)
            configuration: each chunk's randomness is derived from (seed, pass, chunk number) and chunk
            results are merged in a fixed order, independent of OS process scheduling. Requires
            `random_state` to be an explicit integer seed. The seed, worker count, corpus digest and
            per-pass perplexity are recorded in :attr:`training_metadata` and in the log.
            Defaults to False, which preserves the historical (non-deterministic) behavior.
        checkpoint_path : str, optional
            If given, an atomic checkpoint of the model (including the training metadata and corpus
            digest) is written to this path after every pass over the corpus. Training can be resumed
            from the checkpoint with :meth:`resume_from_checkpoint`; with `deterministic=True` the
            resumed result is bit-identical to training in a single run. The previous checkpoint
            generation is kept at ``checkpoint_path + '.prev'`` for partial recovery.

        """
        if workers is not None:
            if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)):
                raise ValueError("workers must be a positive integer, got %r" % (workers,))
            if workers < 1:
                raise ValueError("workers must be a positive integer, got %r" % (workers,))
            if workers > cpu_count():
                logger.warning(
                    "workers=%i exceeds the detected CPU count of %i; this usually hurts performance",
                    workers, cpu_count(),
                )
        self.workers = max(1, cpu_count() - 1) if workers is None else int(workers)
        self.batch = batch

        if isinstance(random_state, (int, np.integer)) and not isinstance(random_state, bool):
            self.random_seed = int(random_state)
        else:
            self.random_seed = None
        if deterministic and self.random_seed is None:
            raise ValueError(
                "deterministic=True requires random_state to be an explicit integer seed, "
                "got %r. The seed is recorded in the model metadata so that runs can be reproduced."
                % (random_state,)
            )
        self.deterministic = bool(deterministic)
        self.checkpoint_path = checkpoint_path
        self._passes_completed = 0
        self._corpus_digest = None
        self._corpus_hasher = None
        from gensim import __version__ as _gensim_version  # lazy import, avoids a circular import
        self.training_metadata = {
            'random_seed': self.random_seed,
            'workers': self.workers,
            'deterministic': self.deterministic,
            'chunksize': chunksize,
            'passes': passes,
            'num_topics': num_topics,
            'corpus_digest': None,
            'vocab_digest': None,
            'per_pass_perplexity': [],
            'gensim_version': _gensim_version,
        }

        if isinstance(alpha, str) and alpha == 'auto':
            raise NotImplementedError("auto-tuning alpha not implemented in LdaMulticore; use plain LdaModel.")

        super(LdaMulticore, self).__init__(
            corpus=corpus, num_topics=num_topics,
            id2word=id2word, chunksize=chunksize, passes=passes, alpha=alpha, eta=eta,
            decay=decay, offset=offset, eval_every=eval_every, iterations=iterations,
            gamma_threshold=gamma_threshold, random_state=random_state, minimum_probability=minimum_probability,
            minimum_phi_value=minimum_phi_value, per_word_topics=per_word_topics, dtype=dtype,
        )

    def update(self, corpus, chunks_as_numpy=False):
        """Train the model with new documents, by EM-iterating over `corpus` until the topics converge
        (or until the maximum number of allowed iterations is reached).

        Train the model with new documents, by EM-iterating over the corpus until the topics converge, or until
        the maximum number of allowed iterations is reached. `corpus` must be an iterable. The E step is distributed
        into the several processes.

        Notes
        -----
        This update also supports updating an already trained model (`self`) with new documents from `corpus`;
        the two models are then merged in proportion to the number of old vs. new documents.
        This feature is still experimental for non-stationary input streams.

        For stationary input (no topic drift in new documents), on the other hand,
        this equals the online update of `'Online Learning for LDA' by Hoffman et al.`_
        and is guaranteed to converge for any `decay` in (0.5, 1].

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`) used to update the
            model.
        chunks_as_numpy : bool
            Whether each chunk passed to the inference step should be a np.ndarray or not. Numpy can in some settings
            turn the term IDs into floats, these will be converted back into integers in inference, which incurs a
            performance hit. For distributed computing it may be desirable to keep the chunks as `numpy.ndarray`.

        """
        try:
            lencorpus = len(corpus)
        except TypeError:
            logger.warning("input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            logger.warning("LdaMulticore.update() called with an empty corpus")
            return

        # In deterministic mode the document count is only accumulated on the first
        # update() call of the training trajectory, so that a run resumed from a
        # checkpoint follows the exact same blending schedule as an uninterrupted run.
        if not self.deterministic or self._passes_completed == 0:
            self.state.numdocs += lencorpus

        if self.batch:
            updatetype = "batch"
            updateafter = lencorpus
        else:
            updatetype = "online"
            updateafter = self.chunksize * self.workers
        eval_every = self.eval_every or 0
        evalafter = min(lencorpus, eval_every * updateafter)

        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info(
            "running %s LDA training, %s topics, %i passes over the supplied corpus of %i documents, "
            "updating every %i documents, evaluating every ~%i documents, "
            "iterating %ix with a convergence threshold of %f",
            updatetype, self.num_topics, self.passes, lencorpus, updateafter,
            evalafter, self.iterations, self.gamma_threshold
        )

        if updates_per_pass * self.passes < 10:
            logger.warning(
                "too few updates, training might not converge; "
                "consider increasing the number of passes or iterations to improve accuracy"
            )

        job_queue = Queue(maxsize=2 * self.workers)
        result_queue = Queue()

        # rho is the "speed" of updating; TODO try other fncs
        # pass_ + num_updates handles increasing the starting t for each pass,
        # while allowing it to "reset" on the first pass of each update
        def rho():
            return pow(self.offset + pass_ + (self.num_updates / self.chunksize), -self.decay)

        deterministic = self.deterministic
        # The corpus digest is tracked in deterministic mode, and whenever checkpointing
        # is enabled (so a later resume can verify the corpus is unchanged).
        track_digest = deterministic or self.checkpoint_path is not None

        if track_digest and self.training_metadata is None:
            # model was pickled before training_metadata existed; rebuild a skeleton
            self.training_metadata = {
                'random_seed': self.random_seed, 'workers': self.workers,
                'deterministic': deterministic, 'chunksize': self.chunksize,
                'passes': self.passes, 'num_topics': self.num_topics,
                'corpus_digest': None, 'vocab_digest': None,
                'per_pass_perplexity': [], 'gensim_version': None,
            }

        if deterministic:
            logger.info(
                "deterministic training: seed=%i, workers=%i, chunksize=%i, corpus digest=%s",
                self.random_seed, self.workers, self.chunksize,
                self._corpus_digest or "<computed during first pass>",
            )
        if track_digest and self._corpus_digest is None and self._corpus_hasher is None:
            # first update() call of this model: start accumulating the corpus digest
            self._corpus_hasher = _new_corpus_hasher(self.id2word)
            self.training_metadata['vocab_digest'] = _vocab_digest_bytes(self.id2word).hex()

        def process_result_queue(force=False):
            """
            Clear the result queue, merging all intermediate results, and update the
            LDA model if necessary.

            """
            merged_new = False
            while not result_queue.empty():
                other.merge(result_queue.get())
                queue_size[0] -= 1
                merged_new = True

            if (force and merged_new and queue_size[0] == 0) or (other.numdocs >= updateafter):
                self.do_mstep(rho(), other, pass_ > 0)
                other.reset()
                if eval_every > 0 and (force or (self.num_updates / updateafter) % eval_every == 0):
                    self.log_perplexity(chunk, total_docs=lencorpus)

        logger.info("training LDA model using %i processes", self.workers)
        pool = Pool(self.workers, worker_e_step, (job_queue, result_queue, self))
        # In deterministic mode, passes are numbered absolutely across update() calls,
        # so that a training resumed from a checkpoint follows the exact same trajectory
        # (same per-chunk seeds, same rho schedule) as an uninterrupted one.
        pass_offset = self._passes_completed if deterministic else 0
        for pass_ in range(pass_offset, pass_offset + self.passes):
            queue_size, reallen = [0], 0
            other = LdaState(self.eta, self.state.sstats.shape)

            if deterministic:
                # Synchronous rounds: dispatch exactly one model update's worth of chunks,
                # wait for all of their results, merge them in chunk order, then update
                # the model. Which state a chunk sees, the M-step schedule and all
                # floating-point summation orders are therefore fully determined by the
                # (seed, workers, chunksize) configuration, not by OS process scheduling.
                chunks_per_round = max(1, -(-updateafter // self.chunksize))
                chunk_iter = iter(utils.grouper(corpus, self.chunksize, as_numpy=chunks_as_numpy))
                chunk_no = 0
                while True:
                    round_chunks = []
                    for _ in range(chunks_per_round):
                        try:
                            round_chunks.append(next(chunk_iter))
                        except StopIteration:
                            break
                    if not round_chunks:
                        break
                    for chunk in round_chunks:
                        reallen += len(chunk)
                        if self._corpus_hasher is not None:
                            _update_corpus_hasher(self._corpus_hasher, chunk)
                        job = (chunk_no, chunk, self.state, _chunk_seed(self.random_seed, pass_, chunk_no))
                        job_queue.put(job)
                        queue_size[0] += 1
                        logger.info(
                            "PROGRESS: pass %i, dispatched chunk #%i = documents up to #%i/%i, "
                            "outstanding queue size %i",
                            pass_, chunk_no, chunk_no * self.chunksize + len(chunk), lencorpus, queue_size[0]
                        )
                        chunk_no += 1
                    round_results = {}
                    for _ in round_chunks:
                        result_chunk_no, result_state = result_queue.get()
                        round_results[result_chunk_no] = result_state
                        queue_size[0] -= 1
                    for result_chunk_no in sorted(round_results):
                        other.merge(round_results[result_chunk_no])
                    self.do_mstep(rho(), other, pass_ > 0)
                    other.reset()
                    if eval_every > 0 and (self.num_updates / updateafter) % eval_every == 0:
                        self.log_perplexity(round_chunks[-1], total_docs=lencorpus)
                # NB: `chunk` still refers to the last chunk of the last round here;
                # it is used for the per-pass evaluation below.
            else:
                chunk_stream = utils.grouper(corpus, self.chunksize, as_numpy=chunks_as_numpy)
                for chunk_no, chunk in enumerate(chunk_stream):
                    reallen += len(chunk)  # keep track of how many documents we've processed so far

                    if self._corpus_hasher is not None:
                        _update_corpus_hasher(self._corpus_hasher, chunk)

                    # put the chunk into the workers' input job queue
                    while True:
                        try:
                            job_queue.put((chunk_no, chunk, self.state), block=False)
                            queue_size[0] += 1
                            logger.info(
                                "PROGRESS: pass %i, dispatched chunk #%i = documents up to #%i/%i, "
                                "outstanding queue size %i",
                                pass_, chunk_no, chunk_no * self.chunksize + len(chunk), lencorpus, queue_size[0]
                            )
                            break
                        except queue.Full:
                            # in case the input job queue is full, keep clearing the
                            # result queue, to make sure we don't deadlock
                            process_result_queue()

                    process_result_queue()
                # endfor single corpus pass

                # wait for all outstanding jobs to finish
                while queue_size[0] > 0:
                    process_result_queue(force=True)

            if reallen != lencorpus:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")

            if deterministic:
                perwordbound = self.log_perplexity(chunk, total_docs=lencorpus)
                perplexity = float(np.exp2(-perwordbound))
                self.training_metadata['per_pass_perplexity'].append(perplexity)
                logger.info(
                    "pass %i/%i finished: per-word bound %.4f, perplexity %.1f",
                    pass_ + 1, pass_offset + self.passes, perwordbound, perplexity,
                )
            if self._corpus_hasher is not None:
                # first pass of the first update() call finished: finalize the corpus digest
                self._corpus_digest = self._corpus_hasher.hexdigest()
                self._corpus_hasher = None
                self.training_metadata['corpus_digest'] = self._corpus_digest
                logger.info("corpus digest: %s", self._corpus_digest)
            if track_digest:
                self._passes_completed = pass_ + 1
                if self.checkpoint_path is not None:
                    self._save_checkpoint(self.checkpoint_path)
        # endfor entire update

        pool.terminate()

    def _save_checkpoint(self, path):
        """Atomically write a training checkpoint to `path`.

        The checkpoint is a single file containing the whole model (including
        `training_metadata`, the corpus digest, the number of completed passes and the
        RNG state), protected by a SHA256 checksum. The file is written to a temporary
        name and moved into place with :func:`os.replace`, so a process killed mid-write
        never leaves a corrupt file behind: the previous generation stays intact, and is
        additionally preserved under ``path + '.prev'``.

        """
        payload = pickle.dumps({'format': _CHECKPOINT_FORMAT, 'model': self}, protocol=4)
        blob = _CHECKPOINT_MAGIC + hashlib.sha256(payload).hexdigest().encode('ascii') + b'\n' + payload
        tmp_path = '%s.tmp-%i' % (path, os.getpid())
        with open(tmp_path, 'wb') as fout:
            fout.write(blob)
            fout.flush()
            os.fsync(fout.fileno())
        if os.path.exists(path):
            os.replace(path, path + '.prev')
        os.replace(tmp_path, path)
        logger.info(
            "checkpoint saved to %s (passes completed: %i, corpus digest: %s)",
            path, self._passes_completed, self._corpus_digest,
        )

    @classmethod
    def _load_checkpoint(cls, path):
        """Load and integrity-check a checkpoint file written by :meth:`_save_checkpoint`.

        Raises
        ------
        CheckpointCorruptError
            If the file is missing, truncated, fails its checksum or contains an
            unexpected payload. The message names the broken part (header, checksum or
            payload) and points to the ``.prev`` file for partial recovery when one exists.

        """
        try:
            with open(path, 'rb') as fin:
                blob = fin.read()
        except OSError as err:
            raise CheckpointCorruptError(
                "cannot read checkpoint file %r: %s. A previous generation may be available at %r."
                % (path, err, path + '.prev')
            )
        if not blob.startswith(_CHECKPOINT_MAGIC):
            raise CheckpointCorruptError(
                "checkpoint file %r has a corrupt header (bad magic bytes): it is not a %s file, "
                "or the header was overwritten. The model payload cannot be recovered from this file; "
                "a previous generation may be available at %r."
                % (path, _CHECKPOINT_FORMAT, path + '.prev')
            )
        hexdigest, sep, payload = blob[len(_CHECKPOINT_MAGIC):].partition(b'\n')
        if not sep:
            raise CheckpointCorruptError(
                "checkpoint file %r is truncated: the checksum line is missing, only the header "
                "could be read. A previous generation may be available at %r." % (path, path + '.prev')
            )
        actual = hashlib.sha256(payload).hexdigest().encode('ascii')
        if actual != hexdigest:
            raise CheckpointCorruptError(
                "checkpoint file %r failed its integrity check: payload checksum mismatch "
                "(expected %s, computed %s). The payload is truncated or was modified on disk. "
                "A previous generation may be available at %r."
                % (path, hexdigest.decode('ascii'), actual.decode('ascii'), path + '.prev')
            )
        try:
            data = pickle.loads(payload)
        except Exception as err:
            raise CheckpointCorruptError(
                "checkpoint file %r has a valid checksum but its payload cannot be unpickled: %s. "
                "This usually means the file was written by an incompatible gensim version. "
                "A previous generation may be available at %r." % (path, err, path + '.prev')
            )
        if not isinstance(data, dict) or data.get('format') != _CHECKPOINT_FORMAT:
            raise CheckpointCorruptError(
                "checkpoint file %r has an unexpected payload format %r; expected %r."
                % (path, data.get('format') if isinstance(data, dict) else type(data), _CHECKPOINT_FORMAT)
            )
        model = data['model']
        if not isinstance(model, LdaMulticore):
            raise CheckpointCorruptError(
                "checkpoint file %r contains a %s, not an LdaMulticore model."
                % (path, type(model).__name__)
            )
        return model

    @classmethod
    def resume_from_checkpoint(cls, path, corpus, total_passes=None):
        """Resume training from a checkpoint written via `checkpoint_path`.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        corpus : iterable of list of (int, float)
            The training corpus. Its digest is recomputed and compared against the digest
            stored in the checkpoint; if the corpus changed since the checkpoint was taken,
            a :class:`CorpusMismatchError` is raised, to avoid mixing incomparable data into
            the model. The corpus must be re-iterable (it is hashed once, then iterated for
            the remaining passes).
        total_passes : int, optional
            Total number of passes the training should reach, including the passes already
            completed before the checkpoint. Defaults to the `passes` value the model was
            constructed with. Use a larger value to extend an already finished training.

        Returns
        -------
        :class:`LdaMulticore`
            The resumed, further-trained model. If the checkpoint was created with
            ``deterministic=True``, the result is bit-identical to an uninterrupted run.

        """
        model = cls._load_checkpoint(path)
        completed = model._passes_completed
        metadata = model.training_metadata or {}
        planned = metadata.get('passes', model.passes) if total_passes is None else total_passes

        if model._corpus_digest is not None:
            actual_digest = corpus_digest(corpus, id2word=model.id2word)
            if actual_digest != model._corpus_digest:
                raise CorpusMismatchError(
                    "the corpus supplied for resuming does not match the checkpoint: "
                    "digest %s != stored %s. Refusing to resume, because training would mix "
                    "incomparable data into the model. Supply the original training corpus."
                    % (actual_digest, model._corpus_digest)
                )
        else:
            logger.warning(
                "checkpoint %r has no corpus digest (created without deterministic mode); "
                "cannot verify that the corpus is unchanged", path,
            )

        remaining = planned - completed
        if remaining <= 0:
            logger.info(
                "checkpoint %r is already fully trained (%i passes); nothing to do", path, completed,
            )
            return model
        logger.info(
            "resuming training from pass %i to %i (%i passes remaining)",
            completed + 1, planned, remaining,
        )
        model.passes = remaining
        if model.training_metadata is not None:
            model.training_metadata['passes'] = planned
        model.update(corpus)
        return model


def worker_e_step(input_queue, result_queue, worker_lda):
    """Perform E-step for each job.

    Parameters
    ----------
    input_queue : queue of (int, list of (int, float), :class:`~gensim.models.ldamodel.LdaState`)
        Each element is a job characterized by its ID, the corpus chunk to be processed in BOW
        format and the LDA state to use for inference. In deterministic mode a fourth element
        is present: the random seed derived for this chunk, which replaces the worker's random
        state so that the result does not depend on which process executes the chunk.
    result_queue : queue of :class:`~gensim.models.ldamodel.LdaState`
        After the worker finished the job, the state of the resulting (trained) worker model is
        appended to this queue. In deterministic mode, results are ``(chunk_no, state)`` tuples
        so the parent process can merge them in chunk order.
    worker_lda : :class:`~gensim.models.ldamulticore.LdaMulticore`
        LDA instance which performed e step
    """
    logger.debug("worker process entering E-step loop")
    while True:
        logger.debug("getting a new job")
        job = input_queue.get()
        if len(job) == 4:
            chunk_no, chunk, w_state, chunk_seed = job
            worker_lda.random_state = np.random.RandomState(chunk_seed)
        else:
            chunk_no, chunk, w_state = job
            chunk_seed = None
        logger.debug("processing chunk #%i of %i documents", chunk_no, len(chunk))
        worker_lda.state = w_state
        worker_lda.sync_state()
        worker_lda.state.reset()
        worker_lda.do_estep(chunk)  # TODO: auto-tune alpha?
        del chunk
        logger.debug("processed chunk, queuing the result")
        if chunk_seed is None:
            result_queue.put(worker_lda.state)
        else:
            result_queue.put((chunk_no, worker_lda.state))
        worker_lda.state = None
        logger.debug("result put")
