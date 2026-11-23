#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""
Automated tests for the keyed (identity-based delete/replace/compact) similarity index,
:mod:`gensim.similarities.keyed`.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import unittest

import numpy

from gensim.similarities import Similarity, KeyedSimilarity
from gensim.test.utils import datapath, get_tmpfile, common_dictionary, common_corpus

logger = logging.getLogger(__name__)

NUM_FEATURES = len(common_dictionary)
DOC_IDS = ['doc%d' % i for i in range(len(common_corpus))]

# test artifacts (recovery logs, concurrency comparisons) land here
OUTPUT_DIR = os.environ.get(
    'GENSIM_TEST_OUTPUT_DIR',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output')),
)


def make_index(prefix=None, num_best=None, keyed=True, doc_ids=None, corpus=None,
               shardsize=3, auto_compact=False, **kwargs):
    if prefix is None:
        prefix = get_tmpfile('keyed_test')
    if corpus is None:
        corpus = common_corpus
    if keyed and doc_ids is None:
        doc_ids = ['doc%d' % i for i in range(len(list(corpus)))]
    return KeyedSimilarity(
        prefix, corpus, num_features=NUM_FEATURES, num_best=num_best, shardsize=shardsize,
        keyed=keyed, doc_ids=doc_ids, auto_compact=auto_compact, **kwargs
    )


class TestKeyedSimilarity(unittest.TestCase):
    """Core delete / replace / query semantics."""

    def test_delete_single_then_no_query_returns_it(self):
        """Acceptance 五.1: after deleting by identity, no query path returns the doc."""
        index = make_index(num_best=3)
        index.delete_document('doc2')
        self.assertEqual(index.live_count, len(DOC_IDS) - 1)

        # path 1: single-document queries, one per indexed document
        for doc in common_corpus:
            result_ids = [doc_id for doc_id, _ in index[doc]]
            self.assertNotIn('doc2', result_ids)

        # path 2: batch query
        for result in index[list(common_corpus)]:
            self.assertNotIn('doc2', [doc_id for doc_id, _ in result])

        # path 3: self-query (iterate the index against itself)
        for result in index:
            self.assertNotIn('doc2', [doc_id for doc_id, _ in result])

        # full-vector path: one column per live document
        index.num_best = None
        full = index[common_corpus[0]]
        self.assertEqual(full.shape, (len(DOC_IDS) - 1,))
        self.assertNotIn('doc2', index.live_doc_ids())

    def test_delete_missing_identity_is_an_explicit_miss(self):
        """Acceptance 三.1: deleting an unknown identity raises, never silently succeeds."""
        index = make_index()
        with self.assertRaises(KeyError):
            index.delete_document('no-such-doc')
        self.assertEqual(index.get_stats()['delete_misses'], 1)
        self.assertEqual(index.live_count, len(DOC_IDS))

    def test_batch_delete_is_atomic(self):
        """Acceptance 一.2: a batch delete either lands completely or not at all."""
        index = make_index(num_best=9)
        with self.assertRaises(KeyError):
            index.delete_documents(['doc1', 'doc2', 'no-such-doc'])
        # nothing was deleted
        self.assertEqual(index.live_count, len(DOC_IDS))
        result_ids = [doc_id for doc_id, _ in index[common_corpus[0]]]
        self.assertIn('doc1', result_ids)
        self.assertIn('doc2', result_ids)

        index.delete_documents(['doc1', 'doc2'])
        self.assertEqual(index.live_count, len(DOC_IDS) - 2)
        result_ids = [doc_id for doc_id, _ in index[common_corpus[0]]]
        self.assertNotIn('doc1', result_ids)
        self.assertNotIn('doc2', result_ids)

    def test_delete_twice_is_a_miss(self):
        index = make_index()
        index.delete_document('doc0')
        with self.assertRaises(KeyError):
            index.delete_document('doc0')

    def test_result_count_not_eaten_by_deletes(self):
        """Acceptance 一.8: asking for N results returns N live documents."""
        # give every document a shared term so all of them have non-zero similarity
        # to the query (num_best always leaves out zero-similarity documents)
        corpus = [[(0, 1.0)] + list(doc) for doc in
                  (common_corpus[i % len(common_corpus)] for i in range(20))]
        doc_ids = ['bulk%d' % i for i in range(20)]
        index = make_index(num_best=10, corpus=corpus, doc_ids=doc_ids, shardsize=7)
        index.delete_documents(doc_ids[::2])  # delete 10 of 20
        result = index[[(0, 1.0)]]
        self.assertEqual(len(result), 10)
        for doc_id, _ in result:
            self.assertNotIn(doc_id, doc_ids[::2])

    def test_replace_keeps_identity_position_and_updates_content(self):
        """Acceptance 一.3 + 五.2: same identity, same position, new content in results."""
        index = make_index(num_best=3)
        pos_before = index.id2pos['doc1']
        query = common_corpus[0]
        sims_before = dict((doc_id, sim) for doc_id, sim in index[query])

        new_content = [(0, 1.0), (10, 1.0), (11, 1.0)]
        index.replace_document('doc1', new_content)

        # identity and position unchanged
        self.assertEqual(index.id2pos['doc1'], pos_before)
        self.assertEqual(index.pos2id[pos_before], 'doc1')
        self.assertEqual(index.live_count, len(DOC_IDS))

        # content updated: similarity of doc1 to the query changed to the exact new value
        sims_after = dict((doc_id, sim) for doc_id, sim in index[query])
        self.assertNotAlmostEqual(sims_before['doc1'], sims_after['doc1'], places=4)
        expected = numpy.dot(
            index.vector_by_key('doc1'),
            numpy.asarray(index._query_vectors(query)[0][0]),
        )
        self.assertAlmostEqual(sims_after['doc1'], expected, places=5)

        # result count not eaten
        self.assertEqual(len(index[query]), 3)

        # full-vector path also reflects the new content
        index.num_best = None
        full = index[query]
        col = index.live_doc_ids().index('doc1')
        self.assertAlmostEqual(full[col], sims_after['doc1'], places=5)

    def test_replace_missing_identity_raises(self):
        index = make_index()
        with self.assertRaises(KeyError):
            index.replace_document('no-such-doc', [(0, 1.0)])

    def test_replace_dimension_mismatch_reports_conflict(self):
        """Acceptance 三.2: dimension conflicts are reported specifically."""
        index = make_index()
        with self.assertRaisesRegex(ValueError, 'dimension mismatch'):
            index.replace_document('doc1', numpy.zeros(NUM_FEATURES + 1, dtype=numpy.float32))
        with self.assertRaisesRegex(ValueError, 'dimension mismatch'):
            index.replace_document('doc1', [(NUM_FEATURES + 5, 1.0)])

    def test_duplicate_identity_registration_reports_conflict(self):
        """Acceptance 三.2: duplicate identity registration is reported specifically."""
        index = make_index()
        with self.assertRaisesRegex(ValueError, 'already registered'):
            index.add_documents([[(0, 1.0)]], doc_ids=['doc0'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            index.add_documents([[(0, 1.0)], [(1, 1.0)]], doc_ids=['new', 'new'])

    def test_deleted_identity_not_reused_within_rounds(self):
        """Acceptance 一.4: a deleted identity cannot be re-registered immediately."""
        index = make_index(id_reuse_rounds=2)
        index.delete_document('doc3')
        with self.assertRaisesRegex(ValueError, 'too recently'):
            index.add_documents([[(0, 1.0)]], doc_ids=['doc3'])
        # after enough mutation rounds, the identity may be registered again
        index.replace_document('doc0', [(0, 1.0)])
        index.replace_document('doc1', [(1, 1.0)])
        index.replace_document('doc2', [(2, 1.0)])
        index.add_documents([[(0, 1.0)]], doc_ids=['doc3'])
        self.assertIn('doc3', index.id2pos)

    def test_stats_are_externally_readable(self):
        """Acceptance 一.9: delete/replace counters are exposed."""
        index = make_index()
        index.delete_document('doc0')
        index.replace_document('doc1', [(0, 1.0)])
        stats = index.get_stats()
        self.assertEqual(stats['deleted'], 1)
        self.assertEqual(stats['replaced'], 1)
        self.assertEqual(stats['added'], len(DOC_IDS))
        self.assertEqual(stats['live'], len(DOC_IDS) - 1)

    def test_keyed_disabled_is_identical_to_plain_similarity(self):
        """Acceptance 一.10: with the feature off, behaviour matches the stock index exactly."""
        query = common_corpus[0]
        for num_best in (None, 3):
            plain = Similarity(
                get_tmpfile('plain'), common_corpus, num_features=NUM_FEATURES,
                num_best=num_best, shardsize=3,
            )
            off = make_index(num_best=num_best, keyed=False, doc_ids=None)
            plain_result = plain[query]
            off_result = off[query]
            if num_best is None:
                self.assertTrue(numpy.allclose(plain_result, off_result))
            else:
                self.assertEqual(len(plain_result), len(off_result))
                for (pos_a, sim_a), (pos_b, sim_b) in zip(plain_result, off_result):
                    self.assertEqual(pos_a, pos_b)
                    self.assertAlmostEqual(sim_a, sim_b, places=6)
            # mutations requiring keyed mode are rejected
            with self.assertRaises(RuntimeError):
                off.delete_document('doc0')
            with self.assertRaises(RuntimeError):
                off.replace_document('doc0', [(0, 1.0)])
            with self.assertRaises(ValueError):
                off.add_documents([[(0, 1.0)]], doc_ids=['x'])

    def test_old_index_loads_then_keyed_layer_enabled(self):
        """Acceptance 二.4: an index persisted in the old format loads unchanged, and the
        keyed layer can be enabled on it afterwards."""
        prefix = get_tmpfile('old_index')
        plain = Similarity(prefix, common_corpus, num_features=NUM_FEATURES, num_best=3,
                           shardsize=3)
        plain.save(prefix)

        # old format loads through the plain class untouched
        loaded_plain = Similarity.load(prefix)
        self.assertFalse(getattr(loaded_plain, 'keyed', False))

        # and through KeyedSimilarity.load with keyed mode still disabled
        loaded = KeyedSimilarity.load(prefix)
        self.assertFalse(getattr(loaded, 'keyed', False))

        # promote: caller supplies identities for the existing documents
        keyed = KeyedSimilarity.promote(loaded_plain, DOC_IDS)
        keyed.delete_document('doc4')
        result_ids = [doc_id for doc_id, _ in keyed[common_corpus[0]]]
        self.assertNotIn('doc4', result_ids)


class TestKeyedPersistence(unittest.TestCase):
    """Crash recovery of deletion state (acceptance 一.6, 五.3)."""

    def _run_subprocess(self, code, env_extra=None):
        env = dict(os.environ)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')
        env.update(env_extra or {})
        return subprocess.run(
            [sys.executable, '-c', code], env=env, cwd=repo_root,
            capture_output=True, text=True, timeout=300,
        )

    def test_delete_state_survives_crash_without_save(self):
        """A process that deletes and then crashes (no save) leaves the deletion on disk."""
        prefix = get_tmpfile('crash_delete')
        code = """
import os
from gensim.similarities import KeyedSimilarity
from gensim.test.utils import common_corpus, common_dictionary
ids = ['doc' + str(i) for i in range(len(common_corpus))]
idx = KeyedSimilarity(%r, common_corpus, num_features=len(common_dictionary),
                      num_best=3, shardsize=3, keyed=True, doc_ids=ids, auto_compact=False)
idx.save(%r)
idx.delete_documents(['doc1', 'doc2'])
os._exit(1)  # simulate power loss: no clean shutdown, no save
""" % (prefix, prefix)
        proc = self._run_subprocess(code)
        self.assertNotEqual(proc.returncode, 0)

        records = []

        class Handler(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = Handler()
        logging.getLogger('gensim.similarities.keyed').addHandler(handler)
        try:
            loaded = KeyedSimilarity.load(prefix)
        finally:
            logging.getLogger('gensim.similarities.keyed').removeHandler(handler)

        self.assertTrue(loaded.keyed)
        self.assertEqual(loaded.live_count, len(DOC_IDS) - 2)
        result_ids = [doc_id for doc_id, _ in loaded[common_corpus[0]]]
        self.assertNotIn('doc1', result_ids)
        self.assertNotIn('doc2', result_ids)
        self.assertEqual(loaded.get_stats()['deleted'], 2)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        log_file = os.path.join(OUTPUT_DIR, 'crash_recovery_delete.log')
        with open(log_file, 'w') as fout:
            fout.write('subprocess exit code: %s\n' % proc.returncode)
            fout.write('recovery log records:\n')
            for record in records:
                fout.write('  %s\n' % record)
        logger.info('recovery log written to %s', log_file)
        self.assertTrue(any('recovered keyed state' in r for r in records))

    def _compaction_crash(self, stage):
        """Crash a subprocess at a given compaction stage; return (prefix, proc, expected)."""
        prefix = get_tmpfile('crash_compact_' + stage)
        code = """
import os
from gensim.similarities import KeyedSimilarity
from gensim.test.utils import common_corpus, common_dictionary
ids = ['doc' + str(i) for i in range(len(common_corpus))]
idx = KeyedSimilarity(%r, common_corpus, num_features=len(common_dictionary),
                      num_best=3, shardsize=3, keyed=True, doc_ids=ids, auto_compact=False)
idx.delete_documents(['doc0', 'doc3', 'doc6'])
idx.replace_document('doc1', [(0, 1.0), (10, 1.0), (11, 1.0)])
idx.save(%r)
idx.compact()
os._exit(0)
""" % (prefix, prefix)
        proc = self._run_subprocess(code, env_extra={'GENSIM_KEYED_CRASH_AT': stage})
        self.assertNotEqual(proc.returncode, 0, msg=proc.stderr[-2000:])
        return prefix, proc

    def _assert_recovered_consistent(self, prefix, proc, stage):
        records = []

        class Handler(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = Handler()
        keyed_logger = logging.getLogger('gensim.similarities.keyed')
        keyed_logger.addHandler(handler)
        try:
            loaded = KeyedSimilarity.load(prefix)
        finally:
            keyed_logger.removeHandler(handler)

        # whichever way recovery went (rollback or roll-forward), the visible state must be
        # exactly the post-delete/replace state
        self.assertEqual(loaded.live_count, len(DOC_IDS) - 3)
        self.assertEqual(sorted(loaded.id2pos), sorted(set(DOC_IDS) - {'doc0', 'doc3', 'doc6'}))
        result = loaded[common_corpus[0]]
        result_ids = [doc_id for doc_id, _ in result]
        for deleted in ('doc0', 'doc3', 'doc6'):
            self.assertNotIn(deleted, result_ids)
        # replacement content is in effect
        sims = dict((doc_id, sim) for doc_id, sim in result)
        self.assertIn('doc1', sims)
        expected = numpy.dot(
            loaded.vector_by_key('doc1'), numpy.asarray(loaded._query_vectors(common_corpus[0])[0][0]),
        )
        self.assertAlmostEqual(sims['doc1'], expected, places=5)
        # no journal or compaction leftovers remain
        self.assertFalse(os.path.exists(prefix + '.compact.journal'))
        self.assertFalse(any(f.startswith(os.path.basename(prefix) + '.compact_tmp')
                             for f in os.listdir(os.path.dirname(prefix))))

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        log_file = os.path.join(OUTPUT_DIR, 'crash_recovery_compact_%s.log' % stage)
        with open(log_file, 'w') as fout:
            fout.write('crash injected at compaction stage: %s\n' % stage)
            fout.write('subprocess exit code: %s\n' % proc.returncode)
            fout.write('recovery log records:\n')
            for record in records:
                fout.write('  %s\n' % record)
        logger.info('recovery log written to %s', log_file)
        self.assertTrue(any('recovered from crash during compaction' in r for r in records),
                        msg='expected a recovery record, got: %r' % (records,))
        return records

    def test_crash_at_compaction_start_rolls_back(self):
        """Acceptance 三.3: crash right after the journal was started -> pre-compaction state."""
        prefix, proc = self._compaction_crash('started')
        self._assert_recovered_consistent(prefix, proc, 'started')

    def test_crash_after_compaction_commit_rolls_forward(self):
        """Acceptance 三.3: crash after the commit point -> compaction completes on load."""
        prefix, proc = self._compaction_crash('committed')
        records = self._assert_recovered_consistent(prefix, proc, 'committed')
        self.assertTrue(any('completed interrupted compaction' in r for r in records))


class TestKeyedCompaction(unittest.TestCase):
    """Compaction: consistency, triggers, failure handling."""

    def test_compaction_preserves_query_results(self):
        """Acceptance 一.5: results are identical before and after compaction."""
        index = make_index(num_best=4)
        index.delete_documents(['doc0', 'doc3', 'doc6'])
        index.replace_document('doc1', [(0, 1.0), (10, 1.0), (11, 1.0)])
        query = common_corpus[2]
        before_topn = index[query]
        index.num_best = None
        before_full = index[query]
        before_ids = index.live_doc_ids()

        self.assertTrue(index.compact())
        self.assertEqual(index.get_stats()['compactions'], 1)
        self.assertEqual(len(index), len(DOC_IDS) - 3)  # physical storage reclaimed

        index.num_best = 4
        after_topn = index[query]
        self.assertEqual([doc_id for doc_id, _ in before_topn],
                         [doc_id for doc_id, _ in after_topn])
        for (_, sim_a), (_, sim_b) in zip(before_topn, after_topn):
            self.assertAlmostEqual(sim_a, sim_b, places=5)

        index.num_best = None
        after_full = index[query]
        after_ids = index.live_doc_ids()
        self.assertEqual(before_ids, after_ids)
        self.assertTrue(numpy.allclose(before_full, after_full, atol=1e-5))

    def test_auto_compaction_triggers_on_ratio(self):
        """Acceptance 一.5: crossing the tombstone ratio compacts asynchronously."""
        index = make_index(num_best=3, tombstone_ratio=0.2, auto_compact=True)
        index.delete_documents(['doc0', 'doc1'])  # 2/9 >= 0.2
        self.assertTrue(index.wait_for_compaction(timeout=60))
        self.assertEqual(index.get_stats()['compactions'], 1)
        self.assertEqual(len(index), len(DOC_IDS) - 2)
        result_ids = [doc_id for doc_id, _ in index[common_corpus[0]]]
        self.assertNotIn('doc0', result_ids)
        self.assertNotIn('doc1', result_ids)

    def test_reclaim_failure_is_observable_and_queries_stay_up(self):
        """Acceptance 三.4: storage failure during reclaim leaves a readable event and a
        fully usable index."""
        index = make_index(num_best=3)
        index.delete_documents(['doc0', 'doc3', 'doc6'])
        index._fail_reclaim = '1'  # test hook: simulate ENOSPC during reclaim
        with self.assertLogs('gensim.similarities.keyed', level='ERROR') as logs:
            self.assertFalse(index.compact())
        self.assertTrue(any('reclaim failed' in message for message in logs.output))
        stats = index.get_stats()
        self.assertEqual(stats['reclaim_failures'], 1)
        self.assertIsNotNone(index.last_reclaim_error)
        # queries unaffected, deletions still in effect (num_best leaves out zero-similarity
        # documents, so exactly the two live non-zero matches come back)
        result_ids = [doc_id for doc_id, _ in index[common_corpus[0]]]
        self.assertEqual(sorted(result_ids), ['doc1', 'doc2'])
        for deleted in ('doc0', 'doc3', 'doc6'):
            self.assertNotIn(deleted, result_ids)
        # and a later retry (storage back) succeeds
        index._fail_reclaim = None
        self.assertTrue(index.compact())
        self.assertEqual(len(index), len(DOC_IDS) - 3)


class TestKeyedConcurrency(unittest.TestCase):
    """Consistent snapshots under concurrent readers + one writer (acceptance 一.7, 五.4)."""

    def test_concurrent_readers_see_consistent_snapshots(self):
        index = make_index(num_best=4)
        query = common_corpus[0]
        batch = ['doc1', 'doc3', 'doc5']

        before = index[query]
        index.delete_documents(batch)
        after = index[query]
        self.assertNotEqual([d for d, _ in before], [d for d, _ in after])

        # rebuild for the concurrent run
        index = make_index(num_best=4)
        stop = threading.Event()
        observations = []
        observations_lock = threading.Lock()
        errors = []

        def reader():
            try:
                while not stop.is_set():
                    result = tuple(index[query])
                    with observations_lock:
                        observations.append(result)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for thread in readers:
            thread.start()

        # two readers query concurrently with no writer active: identical snapshots
        group_a, group_b = [], []
        barrier = threading.Barrier(3)

        def paired_reader(store):
            barrier.wait()
            for _ in range(20):
                store.append(tuple(index[query]))

        ta = threading.Thread(target=paired_reader, args=(group_a,))
        tb = threading.Thread(target=paired_reader, args=(group_b,))
        ta.start()
        tb.start()
        barrier.wait()
        ta.join()
        tb.join()
        self.assertEqual(group_a, group_b)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        compare_file = os.path.join(OUTPUT_DIR, 'concurrent_readers_comparison.json')
        with open(compare_file, 'w') as fout:
            json.dump({
                'reader_a': [[list(pair) for pair in result] for result in group_a[:3]],
                'reader_b': [[list(pair) for pair in result] for result in group_b[:3]],
                'identical': group_a == group_b,
            }, fout, indent=2)

        # writer deletes a batch while readers run; every observed result must be exactly
        # the pre-delete or exactly the post-delete snapshot, never a mixture
        index.delete_documents(batch)
        stop.set()
        for thread in readers:
            thread.join()

        self.assertFalse(errors, msg='readers raised: %r' % (errors,))
        self.assertTrue(observations)
        valid = {tuple(before), tuple(after)}
        for observed in observations:
            self.assertIn(observed, valid)

        with open(compare_file, 'a') as fout:
            fout.write('\nwriter-reader observations: %d total, distinct=%d (expect <= 2)\n'
                       % (len(observations), len(set(observations))))
            fout.write('pre-delete snapshot: %r\n' % (before,))
            fout.write('post-delete snapshot: %r\n' % (after,))


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(module)s:%(lineno)d : %(funcName)s(%(threadName)s) : %(message)s',
        level=logging.DEBUG,
    )
    unittest.main()
