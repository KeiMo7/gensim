#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""
Tests for reproducible (deterministic) training, checkpointing and resuming of
:class:`gensim.models.ldamulticore.LdaMulticore`.
"""

import logging
import os
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import (
    LdaMulticore,
    CheckpointCorruptError,
    CorpusMismatchError,
    corpus_digest,
    compare_topic_word_distributions,
    compare_training_metadata,
)
from gensim.test.utils import get_tmpfile, common_texts

dictionary = Dictionary(common_texts)
corpus = [dictionary.doc2bow(text) for text in common_texts]

logger = logging.getLogger(__name__)


def train_deterministic(corpus, passes=4, workers=2, seed=42, **kwargs):
    params = dict(
        id2word=dictionary, num_topics=5, workers=workers, chunksize=3,
        passes=passes, random_state=seed, deterministic=True,
    )
    params.update(kwargs)
    return LdaMulticore(corpus, **params)


class TestLdaMulticoreDeterministic(unittest.TestCase):
    """Fixed seed + fixed worker count must give bit-identical topic-word distributions."""

    def test_fixed_seed_two_runs_identical(self):
        model_a = train_deterministic(corpus)
        model_b = train_deterministic(corpus)
        # the topic-word distributions must be element-wise identical, not just close
        assert_array_equal(model_a.get_topics(), model_b.get_topics())
        self.assertEqual(
            model_a.training_metadata['corpus_digest'],
            model_b.training_metadata['corpus_digest'],
        )
        self.assertEqual(
            model_a.training_metadata['per_pass_perplexity'],
            model_b.training_metadata['per_pass_perplexity'],
        )

    def test_fixed_seed_identical_across_processes(self):
        # two models trained one after another in this process
        model_a = train_deterministic(corpus)
        # a model trained in a separate Python process, serialized via a regular save
        tmp = get_tmpfile('gensim_lda_deterministic.pkl')
        model_b = train_deterministic(corpus)
        model_b.save(tmp)
        loaded = LdaMulticore.load(tmp)
        assert_array_equal(model_a.get_topics(), loaded.get_topics())

    def test_seed_recorded_in_metadata(self):
        model = train_deterministic(corpus, seed=123)
        self.assertEqual(model.training_metadata['random_seed'], 123)
        self.assertEqual(model.training_metadata['workers'], 2)
        self.assertTrue(model.training_metadata['deterministic'])
        self.assertIsNotNone(model.training_metadata['corpus_digest'])
        self.assertEqual(len(model.training_metadata['per_pass_perplexity']), 4)

    def test_deterministic_requires_explicit_seed(self):
        with self.assertRaises(ValueError):
            LdaMulticore(corpus, id2word=dictionary, num_topics=2, workers=2, deterministic=True)
        # a RandomState object does not expose its seed, so it is rejected too
        with self.assertRaises(ValueError):
            LdaMulticore(
                corpus, id2word=dictionary, num_topics=2, workers=2,
                deterministic=True, random_state=np.random.RandomState(1),
            )

    def test_invalid_workers_rejected_at_startup(self):
        for bad_workers in (0, -1, -10, 2.5, '3', None.__class__):
            with self.assertRaises((ValueError, TypeError)):
                LdaMulticore(
                    corpus, id2word=dictionary, num_topics=2, workers=bad_workers,
                    random_state=1, deterministic=True,
                )

    def test_workers_difference_is_quantified(self):
        model_w1 = train_deterministic(corpus, workers=1)
        model_w2 = train_deterministic(corpus, workers=2)
        model_w2_again = train_deterministic(corpus, workers=2)

        metrics_same = compare_topic_word_distributions(model_w2, model_w2_again)
        # same configuration: no difference at all
        self.assertAlmostEqual(metrics_same['mean_symmetric_kl'], 0.0, places=12)
        self.assertAlmostEqual(metrics_same['min_cosine_similarity'], 1.0, places=12)

        metrics = compare_topic_word_distributions(model_w1, model_w2)
        # different worker counts: the difference is reported as numbers, not adjectives
        for key in ('mean_cosine_similarity', 'min_cosine_similarity',
                    'mean_symmetric_kl', 'max_symmetric_kl', 'mean_top_word_overlap'):
            self.assertTrue(np.isfinite(metrics[key]), key)
        self.assertEqual(len(metrics['topic_mapping']), 5)
        logger.warning(
            "workers=1 vs workers=2: mean cosine similarity %.4f, mean symmetric KL %.4f nats, "
            "mean top-20 word overlap %.2f",
            metrics['mean_cosine_similarity'], metrics['mean_symmetric_kl'],
            metrics['mean_top_word_overlap'],
        )


class TestLdaMulticoreCheckpoint(unittest.TestCase):
    def setUp(self):
        self.checkpoint = get_tmpfile('gensim_lda_checkpoint.ckpt')
        for suffix in ('', '.prev'):
            if os.path.exists(self.checkpoint + suffix):
                os.remove(self.checkpoint + suffix)

    def tearDown(self):
        for suffix in ('', '.prev'):
            if os.path.exists(self.checkpoint + suffix):
                os.remove(self.checkpoint + suffix)

    def test_resume_matches_uninterrupted_training(self):
        one_shot = train_deterministic(corpus, passes=4)

        interrupted = train_deterministic(corpus, passes=2, checkpoint_path=self.checkpoint)
        self.assertEqual(interrupted._passes_completed, 2)
        self.assertTrue(os.path.exists(self.checkpoint))

        resumed = LdaMulticore.resume_from_checkpoint(self.checkpoint, corpus, total_passes=4)
        # resumed training must be element-wise identical to the uninterrupted run
        assert_array_equal(one_shot.get_topics(), resumed.get_topics())
        self.assertEqual(resumed._passes_completed, 4)
        self.assertEqual(
            one_shot.training_metadata['per_pass_perplexity'],
            resumed.training_metadata['per_pass_perplexity'],
        )

    def test_resume_to_original_passes_by_default(self):
        train_deterministic(corpus, passes=2, checkpoint_path=self.checkpoint)
        resumed = LdaMulticore.resume_from_checkpoint(self.checkpoint, corpus)
        # the checkpoint was already fully trained (2 of 2 passes): nothing to do
        self.assertEqual(resumed._passes_completed, 2)

    def test_resume_with_changed_corpus_rejected(self):
        train_deterministic(corpus, passes=2, checkpoint_path=self.checkpoint)
        changed_corpus = [doc[:-1] if doc else doc for doc in corpus]
        with self.assertRaises(CorpusMismatchError):
            LdaMulticore.resume_from_checkpoint(self.checkpoint, changed_corpus, total_passes=4)
        # a truncated corpus is rejected as well
        with self.assertRaises(CorpusMismatchError):
            LdaMulticore.resume_from_checkpoint(self.checkpoint, corpus[:-1], total_passes=4)

    def test_checkpoint_atomic_rotation(self):
        train_deterministic(corpus, passes=3, checkpoint_path=self.checkpoint)
        # after several passes, the previous checkpoint generation is kept for recovery
        self.assertTrue(os.path.exists(self.checkpoint))
        self.assertTrue(os.path.exists(self.checkpoint + '.prev'))
        # no temporary files are left behind
        leftovers = [f for f in os.listdir(os.path.dirname(self.checkpoint))
                     if '.tmp-' in f and f.startswith(os.path.basename(self.checkpoint))]
        self.assertEqual(leftovers, [])

    def test_corrupt_checkpoint_reports_broken_part(self):
        train_deterministic(corpus, passes=2, checkpoint_path=self.checkpoint)

        # corrupt the payload: flip bytes near the end of the file
        with open(self.checkpoint, 'r+b') as fout:
            fout.seek(-16, os.SEEK_END)
            tail = fout.read()
            fout.seek(-16, os.SEEK_END)
            fout.write(bytes(b ^ 0xFF for b in tail))
        with self.assertRaises(CheckpointCorruptError) as ctx:
            LdaMulticore.resume_from_checkpoint(self.checkpoint, corpus, total_passes=4)
        self.assertIn('checksum', str(ctx.exception))
        # the error message points to the recoverable previous generation
        self.assertIn('.prev', str(ctx.exception))

        # corrupt the header
        with open(self.checkpoint, 'r+b') as fout:
            fout.write(b'GARBAGE!!')
        with self.assertRaises(CheckpointCorruptError) as ctx:
            LdaMulticore.resume_from_checkpoint(self.checkpoint, corpus, total_passes=4)
        self.assertIn('header', str(ctx.exception))

        # a missing file is reported as unreadable, with the .prev hint
        with self.assertRaises(CheckpointCorruptError) as ctx:
            LdaMulticore.resume_from_checkpoint(self.checkpoint + '.missing', corpus)
        self.assertIn('.prev', str(ctx.exception))

    def test_model_file_carries_training_metadata(self):
        model = train_deterministic(corpus, passes=2, checkpoint_path=self.checkpoint)
        tmp = get_tmpfile('gensim_lda_metadata.pkl')
        model.save(tmp)
        loaded = LdaMulticore.load(tmp)
        self.assertEqual(loaded.training_metadata['random_seed'], 42)
        self.assertEqual(loaded.training_metadata['corpus_digest'],
                         model.training_metadata['corpus_digest'])
        # two models trained on the same corpus are recognized as comparable
        comparison = compare_training_metadata(model, loaded)
        self.assertTrue(comparison['corpus_digest'][2])
        self.assertTrue(comparison['random_seed'][2])


class TestCorpusDigest(unittest.TestCase):
    def test_digest_is_content_based(self):
        digest_a = corpus_digest(corpus, id2word=dictionary)
        # same content, freshly built list: identical digest, regardless of file origins
        digest_b = corpus_digest([list(doc) for doc in corpus], id2word=dictionary)
        self.assertEqual(digest_a, digest_b)

    def test_digest_changes_with_content(self):
        changed = [doc[:-1] if doc else doc for doc in corpus]
        self.assertNotEqual(corpus_digest(corpus, dictionary), corpus_digest(changed, dictionary))
        reordered = list(reversed(corpus))
        self.assertNotEqual(corpus_digest(corpus, dictionary), corpus_digest(reordered, dictionary))

    def test_digest_covers_vocabulary(self):
        other_dictionary = Dictionary(common_texts + [['extra', 'vocabulary', 'words']])
        self.assertNotEqual(
            corpus_digest(corpus, id2word=dictionary),
            corpus_digest(corpus, id2word=other_dictionary),
        )


class TestBackwardCompatibility(unittest.TestCase):
    def test_old_model_without_new_attributes(self):
        # a model trained/pickled before deterministic mode existed has none of the
        # new attributes; loading and updating it must still work
        model = LdaMulticore(corpus, id2word=dictionary, num_topics=2, passes=1, random_state=1)
        for attr in ('deterministic', 'checkpoint_path', 'random_seed',
                     'training_metadata', '_passes_completed', '_corpus_digest'):
            if attr in model.__dict__:
                delattr(model, attr)
        # falls back to class-level defaults: non-deterministic legacy behavior
        self.assertFalse(model.deterministic)
        model.update(corpus)  # must not raise

    def test_legacy_mode_unaffected_by_new_parameters(self):
        # default construction: deterministic mode off, no checkpointing
        model = LdaMulticore(corpus, id2word=dictionary, num_topics=2, passes=1, random_state=1)
        self.assertFalse(model.deterministic)
        self.assertIsNone(model.checkpoint_path)
        self.assertIsNone(model.training_metadata['corpus_digest'])
        self.assertEqual(model.training_metadata['per_pass_perplexity'], [])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
