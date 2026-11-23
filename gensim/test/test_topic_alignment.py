#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html

"""Tests for cross-batch stable topic numbering (:mod:`gensim.models.topic_alignment`)."""

import json
import logging
import os
import subprocess
import sys
import textwrap
import unittest

import numpy as np

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.topic_alignment import (
    AlignmentConfig,
    AlignmentError,
    DRIFT_NOT_COMPARABLE,
    LedgerError,
    LedgerWriteError,
    TopicAligner,
    TopicLedger,
    TopicSignature,
    align_topics,
    drift_score,
)
from gensim.test.utils import datapath, get_tmpfile
from gensim.utils import simple_preprocess


class _SyntheticModel:
    """A minimal stand-in for a topic model, so merge/split cases are exact.

    It exposes exactly the surface the aligner relies on (``num_topics``,
    ``id2word``, ``get_topics``), which also documents the (small) contract.
    """

    def __init__(self, topics, vocabulary):
        self.num_topics = len(topics)
        self.id2word = {index: word for index, word in enumerate(vocabulary)}
        self._topics = np.asarray(topics, dtype=np.float64)

    def get_topics(self):
        return self._topics


def _one_hot(weights):
    """Build a ``get_topics()`` row from ``{word: weight}`` in a shared vocabulary."""
    return weights


class TestSignatureAndDrift(unittest.TestCase):
    def test_signature_normalises(self):
        signature = TopicSignature([("a", 2.0), ("b", 2.0)], batch_id="b", local_index=0)
        self.assertAlmostEqual(sum(p for _, p in signature.words), 1.0)
        self.assertAlmostEqual(dict(signature.words)["a"], 0.5)

    def test_cosine_is_content_only(self):
        a = TopicSignature([("x", 1.0), ("y", 1.0)], "b1", 0)
        b = TopicSignature([("x", 1.0), ("y", 1.0)], "b2", 5)
        self.assertAlmostEqual(a.cosine(b), 1.0)
        c = TopicSignature([("p", 1.0), ("q", 1.0)], "b2", 0)
        self.assertEqual(a.cosine(c), 0.0)

    def test_drift_zero_for_identical(self):
        a = TopicSignature([("x", 1.0), ("y", 2.0), ("z", 1.0)], "b1", 0)
        b = TopicSignature([("x", 1.0), ("y", 2.0), ("z", 1.0)], "b2", 0)
        score, status = drift_score(a, b, min_shared_terms=3)
        self.assertEqual(status, "ok")
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_drift_not_comparable_never_zero(self):
        a = TopicSignature([("x", 1.0), ("y", 1.0), ("z", 1.0)], "b1", 0)
        b = TopicSignature([("p", 1.0), ("q", 1.0), ("r", 1.0)], "b2", 0)
        score, status = drift_score(a, b, min_shared_terms=3)
        self.assertIsNone(score)
        self.assertEqual(status, DRIFT_NOT_COMPARABLE)

    def test_drift_in_range(self):
        a = TopicSignature([("x", 4.0), ("y", 1.0), ("z", 1.0)], "b1", 0)
        b = TopicSignature([("x", 1.0), ("y", 4.0), ("z", 1.0)], "b2", 0)
        score, status = drift_score(a, b, min_shared_terms=3)
        self.assertEqual(status, "ok")
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestLedgerPersistence(unittest.TestCase):
    def setUp(self):
        self.path = get_tmpfile("topic-ledger-test.json")
        if os.path.exists(self.path):
            os.unlink(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_missing_ledger_reports_path(self):
        with self.assertRaises(LedgerError) as ctx:
            TopicLedger.load(self.path)
        self.assertIn("does not exist", str(ctx.exception))
        self.assertIn(self.path, str(ctx.exception))

    def test_corrupt_ledger_reports_path(self):
        with open(self.path, "w") as handle:
            handle.write("{not json")
        with self.assertRaises(LedgerError) as ctx:
            TopicLedger.load(self.path)
        self.assertIn(self.path, str(ctx.exception))

    def test_wrong_version_rejected(self):
        with open(self.path, "w") as handle:
            json.dump({"format": "gensim-topic-ledger", "version": 99, "batches": [], "next_uid": 1}, handle)
        with self.assertRaises(LedgerError) as ctx:
            TopicLedger.load(self.path)
        self.assertIn("version", str(ctx.exception))

    def test_roundtrip_and_fingerprint(self):
        model = _SyntheticModel([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ["a", "b", "c"])
        config = AlignmentConfig(enabled=True, ledger_path=self.path, topn=3)
        TopicAligner(config).align_and_commit(model, "batch-A")
        loaded = TopicLedger.read_only(self.path)
        self.assertEqual(loaded.batch_ids, ["batch-A"])
        self.assertIn("fingerprint", loaded._payload)
        self.assertEqual(loaded.verify(), [])

    def test_write_failure_raises_and_rolls_back(self):
        config = AlignmentConfig(enabled=True, ledger_path=self.path, topn=3)
        model = _SyntheticModel([[1.0, 0.0]], ["a"])
        aligner = TopicAligner(config)
        result = aligner.align(model, "batch-A")
        # Sabotage the write by pointing at an unwritable location.
        aligner.ledger.path = "/proc/definitely-not-writable/ledger.json"
        with self.assertRaises(LedgerWriteError):
            aligner.commit(result)
        self.assertEqual(aligner.ledger.batch_ids, [])
        self.assertFalse(os.path.exists(self.path))


class TestAlignmentBehaviour(unittest.TestCase):
    """Exact merge / split / disappearance cases using synthetic topic matrices."""

    def _aligner(self, **overrides):
        path = get_tmpfile("topic-ledger-behaviour.json")
        if os.path.exists(path):
            os.unlink(path)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        config = AlignmentConfig(
            enabled=True, ledger_path=path, similarity_threshold=0.30,
            min_shared_terms=2, topn=6, **overrides)
        return TopicAligner(config)

    @staticmethod
    def _model(rows, vocab):
        return _SyntheticModel(rows, vocab)

    def test_continuation_keeps_the_same_identifier(self):
        vocab = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
        aligner = self._aligner()
        first = aligner.align_and_commit(
            self._model([[5, 4, 1, 0, 0, 0], [1, 0, 0, 5, 4, 1]], vocab), "m1")
        second = aligner.align_and_commit(
            self._model([[0, 0, 1, 5, 4, 1], [5, 4, 1, 0, 0, 0]], vocab), "m2")
        # The first topic of m1 is the second topic of m2: same uid, different index.
        self.assertEqual(second.local_to_uid[1], first.local_to_uid[0])
        self.assertEqual(second.local_to_uid[0], first.local_to_uid[1])
        # Drift is computed and comparable for both.
        self.assertEqual(len(second.matches), 2)
        for match in second.matches:
            self.assertIsNotNone(match.drift)
            self.assertEqual(match.drift_status, "ok")

    def test_merge_keeps_primary_and_records_destination(self):
        vocab = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
        aligner = self._aligner()
        first = aligner.align_and_commit(
            self._model([[6, 5, 3, 0, 0, 0], [0, 0, 0, 6, 5, 3]], vocab), "m1")
        # One current topic blends both previous ones -> merge.
        second = aligner.align_and_commit(
            self._model([[6, 5, 3, 6, 5, 3]], vocab), "m2")
        self.assertEqual(len(second.merges), 1)
        merge = second.merges[0]
        self.assertEqual(merge["absorbed"], first.local_to_uid[1])
        self.assertEqual(merge["survivor"], first.local_to_uid[0])
        # The absorbed number is not redistributed to a new local slot.
        self.assertEqual(len(second.local_to_uid), 1)
        # Document mass of the absorbed topic folds into the survivor.
        remapped, report = second.remap_document_topics([(0, 0.5), (1, 0.5)])
        self.assertEqual(remapped, [(0, 1.0)])
        self.assertTrue(report["total_conserved"])

    def test_split_parent_follows_larger_share(self):
        vocab = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
        aligner = self._aligner()
        first = aligner.align_and_commit(
            self._model([[8, 8, 4, 4, 0, 0]], vocab), "m1")
        # Two branches over disjoint words: alpha/beta covers 2/3 of the old mass,
        # gamma/delta the remaining 1/3.
        second = aligner.align_and_commit(
            self._model([[8, 8, 0, 0, 0, 0], [0, 0, 4, 4, 0, 0]], vocab), "m2")
        self.assertEqual(len(second.splits), 1)
        split = second.splits[0]
        self.assertEqual(split["parent"], first.local_to_uid[0])
        self.assertEqual(split["primary_new_local_index"], 0)
        self.assertGreater(split["parent_share"], split["secondary_share"])
        # The secondary branch got a fresh identifier, not the parent's.
        self.assertNotEqual(second.local_to_uid[1], split["parent"])
        self.assertEqual(second.local_to_uid[0], split["parent"])

    def test_split_parent_moves_when_larger_share_is_second(self):
        vocab = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
        aligner = self._aligner()
        first = aligner.align_and_commit(
            self._model([[8, 8, 4, 4, 0, 0]], vocab), "m1")
        # Here the *second* local topic carries the larger share.
        second = aligner.align_and_commit(
            self._model([[0, 0, 4, 4, 0, 0], [8, 8, 0, 0, 0, 0]], vocab), "m2")
        split = second.splits[0]
        self.assertEqual(split["parent"], first.local_to_uid[0])
        self.assertEqual(split["primary_new_local_index"], 1)
        self.assertEqual(second.local_to_uid[1], split["parent"])

    def test_disappearance_has_explicit_destination(self):
        vocab = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta"]
        aligner = self._aligner()
        first = aligner.align_and_commit(
            self._model([[6, 6, 0, 0, 0, 0, 0], [0, 0, 6, 6, 0, 0, 0]], vocab), "m1")
        # The zeta topic vanished; the alpha/beta topic continues.
        second = aligner.align_and_commit(
            self._model([[6, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 7, 7, 7]], vocab), "m2")
        gone = first.local_to_uid[1]
        entry = next(e for e in second.disappearances + second.retirements
                     if e["topic_uid"] == gone)
        self.assertIsNotNone(entry["destination"])
        self.assertIn("folded", entry["reason"])
        # Its document mass is not dropped.
        remapped, report = second.remap_document_topics([(1, 0.7)])
        self.assertAlmostEqual(report["dropped_mass"], 0.0)
        self.assertAlmostEqual(report["output_mass"], 0.7)

    def test_document_remap_conserves_mass(self):
        vocab = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
        aligner = self._aligner()
        aligner.align_and_commit(
            self._model([[6, 6, 0, 0, 0, 0], [0, 0, 6, 6, 0, 0]], vocab), "m1")
        second = aligner.align_and_commit(
            self._model([[6, 6, 0, 0, 0, 0], [0, 0, 6, 6, 0, 0]], vocab), "m2")
        distribution = [(0, 0.25), (1, 0.75)]
        remapped, report = second.remap_document_topics(distribution)
        self.assertTrue(report["total_conserved"])
        self.assertAlmostEqual(sum(weight for _, weight in remapped), 1.0)

    def test_max_topics_absorbs_and_leaves_trace(self):
        vocab = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
        aligner = self._aligner(max_topics=2)
        # Three topics; the third is nearly a duplicate of the first.
        result = aligner.align_and_commit(
            self._model([[6, 6, 0, 0, 0, 0], [0, 0, 6, 6, 0, 0], [6, 5, 0.1, 0, 0, 0]], vocab), "m1")
        self.assertEqual(len(result.local_to_uid), 2)
        self.assertEqual(len(result.absorptions), 1)
        absorption = result.absorptions[0]
        self.assertEqual(absorption["into_new_local_index"], 0)
        self.assertEqual(absorption["absorbed_new_local_index"], 2)

    def test_new_topic_gets_evidence_for_review(self):
        vocab = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
        aligner = self._aligner()
        aligner.align_and_commit(self._model([[6, 6, 0, 0, 0, 0]], vocab), "m1")
        second = aligner.align_and_commit(
            self._model([[6, 6, 0, 0, 0, 0], [0, 0, 0, 7, 7, 7]], vocab), "m2")
        self.assertEqual(len(second.births), 1)
        birth = second.births[0]
        self.assertEqual(second.local_to_uid[1], birth.topic_uid)
        self.assertTrue(birth.label)
        report = second.to_report()
        topic = next(t for t in report["topics"] if t["topic_uid"] == birth.topic_uid)
        self.assertEqual(topic["drift_status"], "new")
        self.assertIn("delta", topic["top_words"])

    def test_conflict_resolution_is_recorded(self):
        vocab = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
        aligner = self._aligner()
        aligner.align_and_commit(
            self._model([[6, 6, 3, 0, 0, 0], [0, 0, 3, 6, 6, 0]], vocab), "m1")
        second = aligner.align_and_commit(
            self._model([[6, 6, 3, 6, 6, 0]], vocab), "m2")
        self.assertGreaterEqual(second.diagnostics["conflicts"]["merge_candidates"], 1)
        self.assertEqual(len(second.merges), 1)
        self.assertIn("reason", second.merges[0])

    def test_disappeared_topic_can_be_revived(self):
        """A topic that vanishes and comes back reclaims its original number."""
        vocab = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
        path = get_tmpfile("topic-ledger-revive.json")
        if os.path.exists(path):
            os.unlink(path)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        config = AlignmentConfig(
            enabled=True, ledger_path=path, similarity_threshold=0.30,
            min_shared_terms=2, topn=6, retired_grace=2)
        aligner = TopicAligner(config)

        first = aligner.align_and_commit(
            self._model([[6, 6, 0, 0, 0, 0], [0, 0, 6, 6, 0, 0]], vocab), "m1")
        vanished_uid = first.local_to_uid[1]
        # Batch 2: the second topic disappears (only the first continues).
        second = aligner.align_and_commit(
            self._model([[6, 6, 0, 0, 0, 0]], vocab), "m2")
        self.assertIn(vanished_uid, {e["topic_uid"] for e in second.disappearances})
        # Batch 3: the vanished content returns — it must reclaim the same identifier.
        third = aligner.align_and_commit(
            self._model([[0, 0, 6, 6, 0, 0], [6, 6, 0, 0, 0, 0]], vocab), "m3")
        self.assertEqual(third.local_to_uid[0], vanished_uid)
        self.assertEqual(len(third.revivals), 1)
        self.assertEqual(third.revivals[0]["topic_uid"], vanished_uid)
        # No fresh identifier was handed to that topic.
        self.assertNotIn(vanished_uid, {b.topic_uid for b in third.births})

    def test_duplicate_batch_id_rejected(self):
        vocab = ["alpha", "beta"]
        aligner = self._aligner()
        aligner.align_and_commit(self._model([[6, 6]], vocab), "same")
        with self.assertRaises(LedgerError) as ctx:
            aligner.align(self._model([[6, 6]], vocab), "same")
        self.assertIn("already recorded", str(ctx.exception))

    def test_require_previous_reports_missing_history(self):
        vocab = ["alpha", "beta"]
        path = get_tmpfile("topic-ledger-require.json")
        if os.path.exists(path):
            os.unlink(path)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        config = AlignmentConfig(enabled=True, ledger_path=path, require_previous=True)
        with self.assertRaises(LedgerError) as ctx:
            TopicAligner(config).align(_SyntheticModel([[1, 1]], vocab), "m1")
        self.assertIn("no previous batch", str(ctx.exception))

    def test_inconsistent_ledger_names_batch_and_counts(self):
        """A ledger whose recorded num_topics disagrees with its active identities is reported."""
        vocab = ["alpha", "beta", "gamma", "delta"]
        aligner = self._aligner()
        aligner.align_and_commit(
            self._model([[6, 6, 0, 0], [0, 0, 6, 6]], vocab), "m1")
        # Corrupt the recorded count, simulating a truncated/edited ledger.
        aligner.ledger._payload["batches"][-1]["num_topics"] = 5
        with self.assertRaises(LedgerError) as ctx:
            aligner.align(self._model([[6, 6, 0, 0], [0, 0, 6, 6]], vocab), "m2")
        message = str(ctx.exception)
        self.assertIn("m1", message)
        self.assertIn("num_topics=5", message)
        self.assertIn("2 identities", message)


class TestDisabledBehaviour(unittest.TestCase):
    def test_disabled_config_makes_no_ledger(self):
        path = get_tmpfile("topic-ledger-disabled.json")
        if os.path.exists(path):
            os.unlink(path)
        config = AlignmentConfig(enabled=False, ledger_path=path)
        result = align_topics(_SyntheticModel([[1.0, 1.0]], ["a", "b"]), config=config)
        self.assertIsNone(result)
        self.assertFalse(os.path.exists(path))

    def test_topic_aligner_refuses_when_disabled(self):
        with self.assertRaises(AlignmentError):
            TopicAligner(AlignmentConfig(enabled=False, ledger_path="unused.json"))


class TestCrossProcessConsistency(unittest.TestCase):
    def test_reader_process_gets_same_identifiers(self):
        path = get_tmpfile("topic-ledger-process.json")
        if os.path.exists(path):
            os.unlink(path)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        vocab = ["alpha", "beta", "gamma", "delta"]
        first = _SyntheticModel([[6, 5, 0, 0], [0, 0, 6, 5]], vocab)
        second = _SyntheticModel([[0, 0, 6, 5], [6, 5, 0, 0]], vocab)
        config = AlignmentConfig(enabled=True, ledger_path=path, topn=4, min_shared_terms=2)
        aligner = TopicAligner(config)
        aligner.align_and_commit(first, "batch-A")
        result_b = aligner.align_and_commit(second, "batch-B")

        script = textwrap.dedent("""
            import json, sys
            sys.path.insert(0, sys.argv[2])
            from gensim.models.topic_alignment import TopicLedger
            ledger = TopicLedger.read_only(sys.argv[1])
            print(json.dumps({
                "batches": ledger.batch_ids,
                "last": sorted(
                    (record["topic_uid"], record["local_index"])
                    for record in ledger.identities_for_batch("batch-B").values()
                    if record.get("local_index") is not None
                ),
                "verify": ledger.verify(),
            }))
        """)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        completed = subprocess.run(
            [sys.executable, "-E", "-c", script, path, repo_root],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["batches"], ["batch-A", "batch-B"])
        self.assertEqual(payload["verify"], [])
        expected = sorted((uid, index) for index, uid in result_b.local_to_uid.items())
        self.assertEqual(payload["last"], [list(item) for item in expected])


class TestEndToEndWithLda(unittest.TestCase):
    """The acceptance scenario: two batches of the repository's own corpus fixtures.

    The Lee corpus is stratified by subject before being interleaved into the two
    batches, so both batches contain every subject. This is what lets the test assert
    the *positive* requirement ("similar topics get the same number") rather than a
    weak lower bound: the same subject must surface in both batches and be matched.
    """

    SUBJECTS = {
        "bushfire": {"fire", "bushfire", "blaze", "winds", "emergency", "homes", "forced",
                     "evacuated", "highlands", "goulburn", "firefighters", "crews"},
        "politics": {"government", "labor", "minister", "election", "parliament", "political",
                     "leader", "party", "howard", "beazley", "ruddock"},
        "middle-east": {"arafat", "palestinian", "israeli", "israel", "qaeda", "afghanistan",
                        "terror", "laden", "hamas", "gaza", "sharon"},
        "aviation": {"qantas", "workers", "airline", "strike", "union", "aircraft", "flight",
                     "engineers", "pay", "industrial"},
    }

    @classmethod
    def _subject(cls, words):
        vocabulary = set(words)
        best, best_hits = None, 0
        for subject, anchors in cls.SUBJECTS.items():
            hits = len(vocabulary & anchors)
            if hits > best_hits:
                best, best_hits = subject, hits
        return best if best_hits >= 2 else None

    @classmethod
    def setUpClass(cls):
        with open(datapath("lee_background.cor"), encoding="latin-1") as handle:
            documents = [[word for word in simple_preprocess(line) if len(word) > 2]
                         for line in handle if line.strip()]
        buckets = {subject: [] for subject in cls.SUBJECTS}
        for document in documents:
            subject = cls._subject(document)
            if subject is not None:
                buckets[subject].append(document)
        cls.texts_a, cls.texts_b = [], []
        for documents in buckets.values():
            for index, document in enumerate(documents):
                (cls.texts_a if index % 2 == 0 else cls.texts_b).append(document)
        cls.dictionary = Dictionary(cls.texts_a + cls.texts_b)
        cls.dictionary.filter_extremes(no_below=2, no_above=0.9)
        cls.corpus_a = [cls.dictionary.doc2bow(text) for text in cls.texts_a]
        cls.corpus_b = [cls.dictionary.doc2bow(text) for text in cls.texts_b]

    def _train(self, corpus, seed, num_topics=5):
        return LdaModel(
            corpus, id2word=self.dictionary, num_topics=num_topics, passes=30,
            random_state=seed, minimum_probability=0.0)

    def test_two_batches_share_identifiers_and_drift(self):
        path = get_tmpfile("topic-ledger-lda.json")
        if os.path.exists(path):
            os.unlink(path)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        config = AlignmentConfig(
            enabled=True, ledger_path=path, topn=50, min_shared_terms=5,
            similarity_threshold=0.30, drift_threshold=0.45)
        aligner = TopicAligner(config)
        model_a = self._train(self.corpus_a, seed=11)
        first = aligner.align_and_commit(model_a, "2026-10")
        model_b = self._train(self.corpus_b, seed=22)
        second = aligner.align_and_commit(model_b, "2026-11")

        # With subject-stratified batches, every topic should find its counterpart.
        self.assertEqual(len(second.matches), 5)
        self.assertEqual(len(second.births), 0)
        # Every match points at a real, current local topic and keeps its uid.
        for match in second.matches:
            self.assertIn(match.new_local_index, second.local_to_uid)
            self.assertEqual(second.local_to_uid[match.new_local_index], match.old.topic_uid)
            self.assertTrue(0.0 <= match.similarity <= 1.0)
            self.assertIsNotNone(match.drift)
            self.assertEqual(match.drift_status, "ok")
        # Every previous topic has an explicit fate.
        covered = {match.old.topic_uid for match in second.matches}
        covered |= {merge["absorbed"] for merge in second.merges}
        covered |= {entry["topic_uid"] for entry in second.disappearances + second.retirements}
        self.assertEqual(covered, set(first.local_to_uid.values()))
        # Document distributions remap with conserved mass.
        original = model_a.get_document_topics(self.corpus_a[0], minimum_probability=0.0)
        remapped, report = second.remap_document_topics(original)
        self.assertTrue(report["total_conserved"])
        self.assertAlmostEqual(sum(w for _, w in remapped), report["input_mass"])
        self.assertEqual(TopicLedger.read_only(path).verify(), [])

    def test_identifier_pairing_is_semantically_plausible(self):
        """Most paired topics should share their subject vocabulary, not just stopwords."""
        path = get_tmpfile("topic-ledger-semantic.json")
        if os.path.exists(path):
            os.unlink(path)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        config = AlignmentConfig(
            enabled=True, ledger_path=path, topn=50, min_shared_terms=5,
            similarity_threshold=0.30)
        aligner = TopicAligner(config)
        model_a = self._train(self.corpus_a, seed=11)
        aligner.align_and_commit(model_a, "a")
        model_b = self._train(self.corpus_b, seed=22)
        second = aligner.align_and_commit(model_b, "b")
        marker_words = set().union(*self.SUBJECTS.values())
        with_subject_evidence = 0
        for match in second.matches:
            previous = {w for w, _ in model_a.show_topic(match.old.local_index, topn=15)}
            current = {w for w, _ in model_b.show_topic(match.new_local_index, topn=15)}
            if (previous & current) & marker_words:
                with_subject_evidence += 1
        # Four of the five subject clusters surface cleanly; the fifth is a generic
        # "south / government / people" topic with no distinctive anchor vocabulary.
        self.assertGreaterEqual(with_subject_evidence, 4)

    def test_incremental_update_keeps_numbers_stable(self):
        path = get_tmpfile("topic-ledger-incremental.json")
        if os.path.exists(path):
            os.unlink(path)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        config = AlignmentConfig(enabled=True, ledger_path=path, topn=50, min_shared_terms=5)
        model = LdaModel(
            self.corpus_a, id2word=self.dictionary, num_topics=5, passes=20,
            random_state=7, minimum_probability=0.0,
            topic_alignment=config, topic_alignment_batch_id="round-1")
        first_uids = dict(model.topic_alignment_result.local_to_uid)
        model.update(self.corpus_b)
        second = model.topic_alignment_result
        self.assertEqual(second.batch_id, "batch-002")
        # Every round-1 uid that survived round 2 is still attached to a topic.
        surviving = {match.old.topic_uid for match in second.matches}
        self.assertTrue(surviving <= set(first_uids.values()))
        for uid in surviving:
            self.assertIn(uid, second.local_to_uid.values())

    def test_existing_topic_output_unchanged_when_disabled(self):
        model = self._train(self.corpus_a, seed=3)
        topics = model.show_topics(num_topics=-1, formatted=False)
        self.assertEqual(len(topics), model.num_topics)
        for topic_id, words in topics:
            self.assertIsInstance(topic_id, int)
            for word, weight in words:
                self.assertIsInstance(word, str)
        self.assertIsNone(model.align_topics())

    def test_legacy_model_without_alignment_config_is_unaffected(self):
        """Loading an old fixture must not require (or run) alignment."""
        legacy = LdaModel.load(datapath("lda_3_0_1_model"))
        self.assertIsNone(legacy.align_topics())
        # Its normal topic output still works exactly as before.
        self.assertEqual(len(legacy.show_topics(num_topics=-1, formatted=False)), legacy.num_topics)

    def test_model_round_trip_preserves_alignment(self):
        path = get_tmpfile("topic-ledger-roundtrip.json")
        model_path = get_tmpfile("topic-aligned-model")
        for candidate in (path, model_path):
            if os.path.exists(candidate):
                os.unlink(candidate)
        self.addCleanup(lambda: [os.path.exists(p) and os.unlink(p) for p in (path, model_path)])
        config = AlignmentConfig(enabled=True, ledger_path=path, topn=50, min_shared_terms=5)
        model = LdaModel(
            self.corpus_a, id2word=self.dictionary, num_topics=5, passes=10,
            random_state=5, minimum_probability=0.0,
            topic_alignment=config, topic_alignment_batch_id="round-1")
        model.save(model_path)
        loaded = LdaModel.load(model_path)
        self.assertTrue(loaded.topic_alignment.enabled)
        self.assertEqual(loaded.topic_alignment_batch_id, "round-1")
        # Aligning a further batch from the reloaded model continues the same ledger.
        result = loaded.align_topics("round-2")
        self.assertEqual(result.batch_id, "round-2")
        self.assertEqual(TopicLedger.read_only(path).batch_ids, ["round-1", "round-2"])


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    unittest.main()
