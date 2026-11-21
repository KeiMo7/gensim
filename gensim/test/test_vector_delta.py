#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Gensim Contributors
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Automated tests for :mod:`gensim.models.vector_delta` (version comparison and
delta-only incremental update) and its command line front-end."""

import json
import os
import unittest
from unittest import mock

import numpy as np

from gensim.models import KeyedVectors
from gensim.models import vector_delta
from gensim.models.vector_delta import (
    BaseVersionMissingError,
    ChangePolicy,
    DimensionMismatchError,
    KeyNormalizer,
    UnsupportedModelError,
    compare_similarity_tasks,
    compare_versions,
    explain_key,
    incremental_update,
    iter_vectors,
    restore_backup,
)
from gensim.scripts import vector_delta as vector_delta_cli
from gensim.test.utils import datapath, get_tmpfile


def _make_kv(keys, vector_size=8, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    kv = KeyedVectors(vector_size, dtype=dtype)
    kv.add_vectors(list(keys), rng.normal(size=(len(keys), vector_size)).astype(dtype))
    for index, key in enumerate(keys):
        kv.set_vecattr(key, 'count', len(keys) - index)
    return kv


class _TempFiles(unittest.TestCase):
    """Give each test a unique set of output paths and clean them up."""

    def setUp(self):
        self._paths = []

    def tmp(self, suffix=''):
        path = get_tmpfile('gensim-vector-delta-test' + suffix)
        if path not in self._paths:
            self._paths.append(path)
        return path

    def tearDown(self):
        for base in self._paths:
            for path in [base] + _related_files(base):
                try:
                    if os.path.isdir(path):
                        import shutil
                        shutil.rmtree(path)
                    elif os.path.exists(path):
                        os.unlink(path)
                except OSError:
                    pass


def _related_files(base):
    directory = os.path.dirname(base) or '.'
    prefix = os.path.basename(base)
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    candidates = [os.path.join(directory, name) for name in names if name.startswith(prefix)]
    return [path for path in candidates if path != base]


class TestKeyNormalizer(unittest.TestCase):
    def test_whitespace_and_case(self):
        normalizer = KeyNormalizer()
        self.assertEqual(normalizer.normalize('  New\tYork '), 'New York')
        self.assertEqual(normalizer.normalize('New  York'), 'New York')
        self.assertEqual(normalizer.normalize('NEW york'), 'NEW york')
        self.assertEqual(KeyNormalizer(case_sensitive=False).normalize('NEW york'), 'new york')

    def test_unicode_normalization(self):
        composed = 'café'
        decomposed = 'café'
        self.assertEqual(KeyNormalizer().normalize(decomposed), composed)
        self.assertNotEqual(KeyNormalizer(unicode_normalization=None).normalize(decomposed), composed)

    def test_non_string_keys_unchanged(self):
        self.assertEqual(KeyNormalizer(case_sensitive=False).normalize(7), 7)

    def test_invalid_unicode_form_rejected(self):
        with self.assertRaises(ValueError):
            KeyNormalizer(unicode_normalization='NFX')


class TestChangePolicy(unittest.TestCase):
    def test_invalid_thresholds_rejected(self):
        with self.assertRaises(vector_delta.ComparisonThresholdError):
            ChangePolicy(relative_l2_threshold=-1)
        with self.assertRaises(vector_delta.ComparisonThresholdError):
            ChangePolicy(cosine_distance_threshold=float('nan'))

    def test_description_documents_metric_and_rationale(self):
        description = ChangePolicy().describe()
        self.assertIn('relative_l2_delta', description['changed_rule'])
        self.assertIn('1.2e-7', description['rationale'])


class TestCompareVersions(_TempFiles):
    def setUp(self):
        super().setUp()
        self.keys = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        self.old = _make_kv(self.keys, seed=1)
        self.old_path = self.tmp('.old.kv')
        self.old.save(self.old_path)

    def _new_version(self, mutate=None):
        new = KeyedVectors.load(self.old_path)
        if mutate is not None:
            mutate(new)
        return new

    def test_vocabulary_change_counts(self):
        """Added, removed and common keys are counted separately."""
        def mutate(new):
            new.add_vector('zeta', np.ones(8, dtype=np.float32))
            new.add_vector('eta', np.full(8, 2.0, dtype=np.float32))
            keep = [i for i, key in enumerate(new.index_to_key) if key != 'delta']
            new.index_to_key = [new.index_to_key[i] for i in keep]
            new.vectors = new.vectors[keep]
            new.key_to_index = {key: i for i, key in enumerate(new.index_to_key)}
            for attr, values in list(new.expandos.items()):
                new.expandos[attr] = values[keep]

        new_path = self.tmp('.new.kv')
        self._new_version(mutate).save(new_path)
        report = compare_versions(self.old_path, new_path, streaming=True)

        self.assertTrue(report.compatible)
        self.assertEqual(report.vocabulary.old_keys, 5)
        self.assertEqual(report.vocabulary.new_keys, 6)
        self.assertEqual(report.vocabulary.added, 2)
        self.assertEqual(report.vocabulary.removed, 1)
        self.assertEqual(report.vocabulary.common, 4)
        self.assertEqual(report.vocabulary.changed, 0)
        self.assertEqual(report.vocabulary.unchanged_exact, 4)
        self.assertIn('zeta', report.vocabulary.added_examples)
        self.assertIn('delta', report.vocabulary.removed_examples)

    def test_vector_change_counts_and_metrics(self):
        """Vectors beyond the threshold count as changed; tiny perturbations stay in the grey zone."""
        def mutate(new):
            new.vectors[0] += 1.0  # alpha: clearly changed
            new.vectors[1] += 1e-9  # beta: below any reasonable threshold
            new.vectors[2] += 1e-8  # gamma: still within threshold

        new_path = self.tmp('.new.kv')
        self._new_version(mutate).save(new_path)
        report = compare_versions(self.old_path, new_path)

        self.assertEqual(report.vocabulary.changed, 1)
        self.assertEqual(report.vocabulary.unchanged_exact, 2)
        self.assertEqual(report.vocabulary.unchanged_within_threshold, 2)
        self.assertEqual(report.vectors.compared, 5)
        self.assertGreater(report.vectors.max_relative_l2_delta, 0.1)
        self.assertIsNotNone(report.vectors.relative_l2_percentiles)
        example_keys = [example['key'] for example in report.vectors.changed_examples]
        self.assertEqual(example_keys, ['alpha'])

    def test_threshold_is_configurable(self):
        def mutate(new):
            new.vectors[0] += 0.01

        new_path = self.tmp('.new.kv')
        self._new_version(mutate).save(new_path)
        strict = compare_versions(
            self.old_path, new_path, policy=ChangePolicy(relative_l2_threshold=0.0, cosine_distance_threshold=0.0),
        )
        loose = compare_versions(self.old_path, new_path, policy=ChangePolicy(relative_l2_threshold=1e9,
                                                                               cosine_distance_threshold=2.0))
        self.assertEqual(strict.vocabulary.changed, 1)
        self.assertEqual(loose.vocabulary.changed, 0)

    def test_streaming_and_in_memory_agree(self):
        def mutate(new):
            new.vectors[[0, 3]] += 0.5
            new.add_vector('zeta', np.ones(8, dtype=np.float32))

        new = self._new_version(mutate)
        streaming = compare_versions(self.old_path, new, streaming=True)
        memory = compare_versions(self.old, new, streaming=False)
        for field in (
            'old_keys', 'new_keys', 'added', 'removed', 'common', 'changed',
            'unchanged_exact', 'unchanged_within_threshold',
        ):
            self.assertEqual(getattr(streaming.vocabulary, field), getattr(memory.vocabulary, field), field)

    def test_reports_are_written(self):
        report = compare_versions(self.old, self.old)
        json_path = self.tmp('.json')
        markdown_path = self.tmp('.md')
        report.write_json(json_path)
        report.write_markdown(markdown_path)
        with open(json_path, encoding='utf8') as fin:
            data = json.load(fin)
        self.assertEqual(data['schema_version'], vector_delta.SCHEMA_VERSION)
        self.assertEqual(data['vocabulary']['changed'], 0)
        self.assertIn('Metric and thresholds', open(markdown_path, encoding='utf8').read())

    def test_similarity_task_comparison(self):
        old = _make_kv(['love', 'sex', 'tiger', 'cat', 'book', 'paper', 'computer', 'keyboard'], seed=3)
        new = _make_kv(['love', 'sex', 'tiger', 'cat', 'book', 'paper', 'computer', 'keyboard'], seed=3)
        new.vectors[0] += 5.0
        comparison = compare_similarity_tasks(old, new, pairs=datapath('wordsim353.tsv'))
        self.assertIn('pearson', comparison['old']['word_pairs'])
        self.assertIn('pearson', comparison['new']['word_pairs'])
        self.assertNotEqual(comparison['delta']['word_pairs']['pearson'], 0.0)


class TestDimensionMismatch(_TempFiles):
    def test_compare_refuses_dimension_mismatch(self):
        old = _make_kv(['a', 'b', 'c'], vector_size=8, seed=5)
        new = _make_kv(['a', 'b', 'c'], vector_size=16, seed=5)
        with self.assertRaises(DimensionMismatchError) as caught:
            compare_versions(old, new)
        message = str(caught.exception)
        self.assertIn('different vector sizes', message)
        self.assertIn('dim 8', message)
        self.assertIn('dim 16', message)
        self.assertIn('not comparable', message)

    def test_explain_refuses_dimension_mismatch(self):
        old = _make_kv(['a', 'b'], vector_size=8)
        new = _make_kv(['a', 'b'], vector_size=4)
        with self.assertRaises(DimensionMismatchError):
            explain_key(old, new, 'a')

    def test_update_refuses_dimension_mismatch(self):
        base = _make_kv(['a', 'b'], vector_size=8)
        updates = {'a': np.ones(4, dtype=np.float32)}
        with self.assertRaises(DimensionMismatchError):
            incremental_update(base, updates, self.tmp('.out.kv'))

    def test_dimension_mismatch_is_a_value_error(self):
        self.assertTrue(issubclass(DimensionMismatchError, ValueError))


class TestExplainKey(_TempFiles):
    def setUp(self):
        super().setUp()
        self.keys = ['king', 'queen', 'man', 'woman', 'apple']
        self.old = _make_kv(self.keys, seed=11)
        self.new = _make_kv(self.keys, seed=11)
        self.new.vectors[0] = self.new.vectors[0] + np.array([0.0, 0.0, 5.0] + [0.0] * 5, dtype=np.float32)

    def test_offset_and_neighbor_change(self):
        delta = explain_key(self.old, self.new, 'king', topn=3)
        self.assertEqual(delta.status, 'changed')
        self.assertGreater(delta.relative_l2_delta, 0.1)
        self.assertIsNotNone(delta.cosine_distance)
        self.assertEqual(len(delta.old_nearest_neighbors), 3)
        self.assertEqual(len(delta.new_nearest_neighbors), 3)
        self.assertIsNotNone(delta.nearest_neighbor_changed)
        self.assertIsNotNone(delta.nearest_neighbor_overlap)

    def test_unchanged_key_is_exact(self):
        delta = explain_key(self.old, self.new, 'queen', topn=2)
        self.assertEqual(delta.status, 'unchanged_exact')
        self.assertEqual(delta.l2_delta, 0.0)

    def test_added_and_removed_keys(self):
        new = _make_kv(['king', 'queen', 'man', 'woman', 'pear'], seed=11)
        self.assertEqual(explain_key(self.old, new, 'apple', topn=0).status, 'removed')
        self.assertEqual(explain_key(self.old, new, 'pear', topn=0).status, 'added')

    def test_lookup_uses_normalized_identity(self):
        delta = explain_key(self.old, self.new, '  KING ', normalizer=KeyNormalizer(case_sensitive=False), topn=0)
        self.assertEqual(delta.old_spelling, 'king')
        self.assertEqual(delta.status, 'changed')

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            explain_key(self.old, self.new, 'not-there', topn=0)


class TestIncrementalUpdate(_TempFiles):
    def setUp(self):
        super().setUp()
        self.keys = ['w%d' % i for i in range(10)]
        self.base = _make_kv(self.keys, seed=21)
        self.base_path = self.tmp('.base.kv')
        self.base.save(self.base_path)

    def test_unchanged_rows_are_bitwise_identical(self):
        """Acceptance: unchanged words keep their exact bytes after an incremental update."""
        changed = np.arange(8, dtype=np.float32) * 0.5
        updates = {'w3': changed, 'w_new': np.full(8, 9.0, dtype=np.float32)}
        output = self.tmp('.out.kv')
        record = incremental_update(self.base_path, updates, output)

        updated = KeyedVectors.load(output)
        self.assertEqual(record.rows_replaced, 1)
        self.assertEqual(record.rows_added, 1)
        self.assertEqual(len(updated), 11)
        for key in self.keys:
            if key == 'w3':
                continue
            old_bytes = self.base.get_vector(key).tobytes()
            new_bytes = updated.get_vector(key).tobytes()
            self.assertEqual(old_bytes, new_bytes, key)
        np.testing.assert_array_equal(updated.get_vector('w3'), changed)
        np.testing.assert_array_equal(updated.get_vector('w_new'), np.full(8, 9.0, dtype=np.float32))

    def test_identical_update_does_not_replace(self):
        updates = {'w1': self.base.get_vector('w1')}
        output = self.tmp('.out.kv')
        record = incremental_update(self.base_path, updates, output)
        self.assertEqual(record.rows_replaced, 0)
        self.assertEqual(record.rows_reused_exact, 1)
        self.assertEqual(record.rows_unchanged_kept, 9)

    def test_update_preserves_expandos(self):
        updates = {'w2': self.base.get_vector('w2') + np.float32(0.25)}
        output = self.tmp('.out.kv')
        incremental_update(self.base_path, updates, output)
        updated = KeyedVectors.load(output)
        self.assertEqual(updated.get_vecattr('w2', 'count'), self.base.get_vecattr('w2', 'count'))
        self.assertEqual(updated.get_vecattr('w9', 'count'), self.base.get_vecattr('w9', 'count'))

    def test_prune_missing(self):
        output = self.tmp('.out.kv')
        record = incremental_update(self.base_path, {'w0': self.base.get_vector('w0') + np.float32(1)},
                                    output, prune_missing=True)
        updated = KeyedVectors.load(output)
        self.assertEqual(list(updated.index_to_key), ['w0'])
        self.assertEqual(record.rows_pruned, 9)

    def test_missing_base_is_refused(self):
        with self.assertRaises(BaseVersionMissingError) as caught:
            incremental_update(self.tmp('.does-not-exist.kv'), {'w0': np.ones(8)}, self.tmp('.out.kv'))
        self.assertIn('rebuild', str(caught.exception))

    def test_fasttext_vectors_are_refused(self):
        from gensim.models.fasttext import FastTextKeyedVectors
        ft = FastTextKeyedVectors(vector_size=8, min_n=3, max_n=5, bucket=1000, count=1)
        ft.add_vector('word', np.ones(8, dtype=np.float32))
        with self.assertRaises(UnsupportedModelError):
            incremental_update(ft, {'word': np.zeros(8)}, self.tmp('.out.kv'))

    def test_existing_output_requires_overwrite(self):
        output = self.tmp('.out.kv')
        incremental_update(self.base_path, {}, output)
        with self.assertRaises(FileExistsError):
            incremental_update(self.base_path, {}, output)

    def test_backup_and_restore(self):
        output = self.tmp('.out.kv')
        incremental_update(self.base_path, {'w0': self.base.get_vector('w0') + np.float32(1)}, output)
        original_bytes = KeyedVectors.load(output).get_vector('w0').tobytes()
        backup_dir = self.tmp('.backup')
        record = incremental_update(
            output, {'w0': np.zeros(8, dtype=np.float32)}, output, overwrite=True, backup_dir=backup_dir,
        )
        self.assertEqual(record.backup_dir, os.path.abspath(backup_dir))
        self.assertNotEqual(KeyedVectors.load(output).get_vector('w0').tobytes(), original_bytes)
        restore_backup(output, backup_dir)
        self.assertEqual(KeyedVectors.load(output).get_vector('w0').tobytes(), original_bytes)

    def test_failure_mid_update_keeps_original_model(self):
        """Acceptance: a failure while saving leaves the previous model untouched."""
        output = self.tmp('.out.kv')
        incremental_update(self.base_path, {'w0': self.base.get_vector('w0') + np.float32(1)}, output)
        before = KeyedVectors.load(output)
        before_bytes = [before.get_vector(key).tobytes() for key in before.index_to_key]

        real_save = KeyedVectors.save

        def exploding_save(self, *args, **kwargs):
            real_save(self, *args, **kwargs)
            raise IOError('simulated disk failure after a partial write')

        with mock.patch.object(KeyedVectors, 'save', exploding_save):
            with self.assertRaises(IOError):
                incremental_update(
                    output, {'w0': np.zeros(8, dtype=np.float32), 'w_new': np.ones(8, dtype=np.float32)},
                    output, overwrite=True, backup_dir=self.tmp('.backup'),
                )

        after = KeyedVectors.load(output)
        self.assertEqual(after.index_to_key, before.index_to_key)
        self.assertEqual([after.get_vector(key).tobytes() for key in after.index_to_key], before_bytes)
        self.assertFalse(
            [name for name in os.listdir(os.path.dirname(output)) if name.startswith('.' + os.path.basename(output))],
            'temporary files must be cleaned up',
        )

    def test_update_record_json(self):
        output = self.tmp('.out.kv')
        record = incremental_update(self.base_path, {'w1': self.base.get_vector('w1') + np.float32(2)}, output)
        path = self.tmp('.record.json')
        record.write_json(path)
        with open(path, encoding='utf8') as fin:
            data = json.load(fin)
        self.assertEqual(data['rows_total'], 10)
        self.assertEqual(data['rows_replaced'], 1)
        self.assertTrue(data['completed'])
        self.assertEqual(data['changed_keys'][0]['key'], 'w1')
        self.assertIn('relative_l2_delta', data['policy']['changed_rule'])


class TestSourcesAndStreaming(_TempFiles):
    def test_word2vec_text_roundtrip_streaming(self):
        keys = ['k%d' % i for i in range(20)]
        old = _make_kv(keys, seed=31)
        new = _make_kv(keys, seed=31)
        new.vectors[4] += 1.0
        old_txt = self.tmp('.old.txt')
        new_txt = self.tmp('.new.txt')
        old.save_word2vec_format(old_txt, binary=False)
        new.save_word2vec_format(new_txt, binary=False)

        report = compare_versions(old_txt, new_txt, format='word2vec-text', streaming=True)
        self.assertEqual(report.vocabulary.changed, 1)
        self.assertEqual(report.vocabulary.common, 20)

    def test_word2vec_binary_streaming_matches_object_comparison(self):
        keys = ['k%d' % i for i in range(50)]
        old = _make_kv(keys, seed=41)
        new = _make_kv(keys, seed=41)
        new.vectors[7] += np.float32(3.0)
        old_bin = self.tmp('.old.bin')
        new_bin = self.tmp('.new.bin')
        old.save_word2vec_format(old_bin, binary=True)
        new.save_word2vec_format(new_bin, binary=True)

        from_files = compare_versions(old_bin, new_bin, format='word2vec-binary', streaming=True)
        from_objects = compare_versions(old, new, streaming=False)
        self.assertEqual(from_files.vocabulary.changed, from_objects.vocabulary.changed)
        self.assertEqual(from_files.vocabulary.common, from_objects.vocabulary.common)
        self.assertIn('k7', [example['key'] for example in from_files.vectors.changed_examples])

    def test_iter_vectors_chunks_respect_size(self):
        kv = _make_kv(['a', 'b', 'c', 'd', 'e'], seed=51)
        chunks = list(iter_vectors(kv, chunk_size=2))
        self.assertEqual([len(keys) for keys, _ in chunks], [2, 2, 1])

    def test_normalization_collision_is_counted(self):
        old = _make_kv(['word', 'Word'], seed=61)
        report = compare_versions(old, old, normalizer=KeyNormalizer(case_sensitive=False), streaming=False)
        self.assertEqual(report.vocabulary.old_normalization_collisions, 1)
        self.assertEqual(report.vocabulary.common, 1)


class TestCommandLine(_TempFiles):
    def setUp(self):
        super().setUp()
        self.old = _make_kv(['a', 'b', 'c'], seed=71)
        self.old_path = self.tmp('.old.kv')
        self.old.save(self.old_path)
        self.new = _make_kv(['a', 'b', 'c', 'd'], seed=71)
        self.new_path = self.tmp('.new.kv')
        self.new.save(self.new_path)

    def test_compare_command(self):
        report_json = self.tmp('.report.json')
        report_md = self.tmp('.report.md')
        code = vector_delta_cli.main([
            'compare', self.old_path, self.new_path, '--report-json', report_json,
            '--report-markdown', report_md, '--max-examples', '2',
        ])
        self.assertEqual(code, 0)
        with open(report_json, encoding='utf8') as fin:
            data = json.load(fin)
        self.assertEqual(data['vocabulary']['added'], 1)
        self.assertTrue(os.path.exists(report_md))

    def test_update_and_restore_commands(self):
        output = self.tmp('.out.kv')
        record_path = self.tmp('.record.json')
        backup_dir = self.tmp('.backup')
        code = vector_delta_cli.main(['update', self.old_path, self.new_path, output, '--record', record_path])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(record_path))
        with open(record_path, encoding='utf8') as fin:
            record = json.load(fin)
        self.assertEqual(record['rows_added'], 1)

        # Replace the output, keeping a backup, then roll back to it.
        before = KeyedVectors.load(output)
        before_bytes = [before.get_vector(key).tobytes() for key in before.index_to_key]
        code = vector_delta_cli.main([
            'update', self.old_path, self.old_path, output, '--overwrite', '--backup-dir', backup_dir,
            '--prune-missing',
        ])
        self.assertEqual(code, 0)
        self.assertEqual(
            vector_delta_cli.main(['restore', output, '--backup-dir', backup_dir, '--remove-backup']), 0,
        )
        restored = KeyedVectors.load(output)
        self.assertEqual([restored.get_vector(key).tobytes() for key in restored.index_to_key], before_bytes)
        self.assertFalse(os.path.exists(backup_dir))

    def test_dimension_mismatch_exit_code(self):
        other = _make_kv(['a', 'b'], vector_size=4, seed=72)
        other_path = self.tmp('.other.kv')
        other.save(other_path)
        self.assertEqual(vector_delta_cli.main(['compare', self.old_path, other_path]), 2)

    def test_explain_command(self):
        code = vector_delta_cli.main(['explain', self.old_path, self.new_path, '--key', 'a', '--topn', '1'])
        self.assertEqual(code, 0)


class TestExistingBehaviorUnchanged(unittest.TestCase):
    """The new capability is opt-in: the old APIs behave exactly as before."""

    def test_training_and_similarity_api_unchanged(self):
        from gensim.models import Word2Vec
        from gensim.test.utils import common_texts
        model = Word2Vec(common_texts, vector_size=24, window=5, min_count=1, seed=42, workers=1,
                         sample=0.0, negative=5, epochs=5)
        # Same public API as before; no new required arguments.
        model.wv.most_similar('human', topn=3)
        model.wv.similarity('human', 'computer')
        model.wv.distances('human')
        self.assertFalse(hasattr(model.wv, '_vector_delta_enabled'))

    def test_keyedvectors_save_load_unchanged(self):
        kv = _make_kv(['a', 'b'], seed=81)
        path = get_tmpfile('gensim-vector-delta-existing.kv')
        try:
            kv.save(path, separately=[])
            loaded = KeyedVectors.load(path)
            np.testing.assert_array_equal(loaded.vectors, kv.vectors)
            self.assertEqual(loaded.index_to_key, kv.index_to_key)
        finally:
            for suffix in ('', '.npy', '.vectors.npy'):
                try:
                    os.unlink(path + suffix)
                except OSError:
                    pass


if __name__ == '__main__':
    unittest.main()
