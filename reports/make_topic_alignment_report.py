#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Acceptance run for cross-batch stable topic numbering.

Produces ``reports/topic-alignment-report.json`` and ``reports/topic-alignment-report.md``
from the repository's own LDA fixtures (``lee_background.cor`` split into two batches).

The report contains topic identifiers, words and counts only — never raw document
content — so it can be shared with business stakeholders.

Usage::

    python -I reports/make_topic_alignment_report.py
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile

import numpy as np

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.topic_alignment import AlignmentConfig, TopicAligner, TopicLedger
from gensim.parsing.preprocessing import STOPWORDS
from gensim.test.utils import datapath
from gensim.utils import simple_preprocess

NUM_TOPICS = 5

# Vocabulary that anchors each news subject in the Lee corpus. Documents are
# stratified by subject *before* being interleaved into two batches, so each batch
# contains the same mixture of subjects. Without this, the corpus is dominated by
# stopword-like words (``will``, ``been``, ``said``) and every topic looks alike.
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


def subject_of(words):
    """Return the subject whose anchor vocabulary best matches ``words`` (or None)."""
    vocabulary = set(words)
    best, best_hits = None, 0
    for subject, anchors in SUBJECTS.items():
        hits = len(vocabulary & anchors)
        if hits > best_hits:
            best, best_hits = subject, hits
    return best if best_hits >= 2 else None


def load_batches():
    """Split the repository's Lee corpus into two subject-stratified batches.

    Documents are grouped by subject, then alternately assigned to batch A / batch B,
    so both batches contain every subject in roughly equal proportion.
    """
    with open(datapath("lee_background.cor"), encoding="latin-1") as handle:
        documents = [[w for w in simple_preprocess(line) if len(w) > 2 and w not in STOPWORDS]
                     for line in handle if line.strip()]
    buckets = {subject: [] for subject in SUBJECTS}
    for document in documents:
        subject = subject_of(document)
        if subject is not None:
            buckets[subject].append(document)
    batch_a, batch_b = [], []
    for documents in buckets.values():
        for index, document in enumerate(documents):
            (batch_a if index % 2 == 0 else batch_b).append(document)
    return batch_a, batch_b


def train(corpus, dictionary, seed):
    return LdaModel(
        corpus, id2word=dictionary, num_topics=NUM_TOPICS, passes=30,
        random_state=seed, minimum_probability=0.0)


def topic_words(model, topic_id, topn=8):
    return [word for word, _ in model.show_topic(topic_id, topn=topn)]


def cross_process_check(ledger_path, expected):
    """Read the ledger from a fresh interpreter and confirm identical identifiers."""
    script = (
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from gensim.models.topic_alignment import TopicLedger\n"
        "ledger = TopicLedger.read_only(sys.argv[1])\n"
        "print(json.dumps({\n"
        "  'batches': ledger.batch_ids,\n"
        "  'last': sorted((r['topic_uid'], r['local_index'])\n"
        "                 for r in ledger.identities_for_batch(sys.argv[3]).values()\n"
        "                 if r.get('local_index') is not None),\n"
        "  'verify': ledger.verify(),\n"
        "}))\n"
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    completed = subprocess.run(
        [sys.executable, "-E", "-c", script, ledger_path, repo_root,
         expected["batch_id"]],
        capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise RuntimeError("cross-process read failed:\n" + completed.stderr)
    payload = json.loads(completed.stdout)
    payload["matches_in_process"] = payload["last"] == sorted(
        [uid, index] for index, uid in expected["local_to_uid"].items())
    return payload


def synthetic_fate_cases():
    """Exact merge / split / disappearance / max_topics cases on tiny matrices."""
    vocabulary = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]

    class Model:
        def __init__(self, rows):
            self.num_topics = len(rows)
            self.id2word = dict(enumerate(vocabulary))
            self._topics = np.asarray(rows, dtype=np.float64)

        def get_topics(self):
            return self._topics

    cases = {}
    with tempfile.TemporaryDirectory() as tmp:
        config = AlignmentConfig(
            enabled=True, ledger_path=os.path.join(tmp, "fates.json"),
            similarity_threshold=0.30, min_shared_terms=2, topn=6)
        aligner = TopicAligner(config)

        # Merge: two previous topics collapse into one current topic.
        first = aligner.align_and_commit(
            Model([[6, 5, 3, 0, 0, 0], [0, 0, 0, 6, 5, 3]]), "merge-1")
        second = aligner.align_and_commit(Model([[6, 5, 3, 6, 5, 3]]), "merge-2")
        cases["merge"] = {
            "previous": first.local_to_uid,
            "current": second.local_to_uid,
            "merges": second.merges,
            "remap_of_both_previous_topics": second.remap_document_topics([(0, 0.5), (1, 0.5)]),
        }

        # Split: one previous topic spreads into two branches of unequal mass.
        first = aligner.align_and_commit(Model([[8, 8, 4, 4, 0, 0]]), "split-1")
        second = aligner.align_and_commit(
            Model([[0, 0, 4, 4, 0, 0], [8, 8, 0, 0, 0, 0]]), "split-2")
        cases["split"] = {
            "previous": first.local_to_uid,
            "current": second.local_to_uid,
            "splits": second.splits,
        }

        # Disappearance: a previous topic has no successor.
        first = aligner.align_and_commit(
            Model([[6, 6, 0, 0, 0, 0], [0, 0, 6, 6, 0, 0]]), "gone-1")
        second = aligner.align_and_commit(
            Model([[6, 6, 0, 0, 0, 0], [0, 0, 0, 0, 7, 7]]), "gone-2")
        cases["disappearance"] = {
            "previous": first.local_to_uid,
            "current": second.local_to_uid,
            "disappearances": second.disappearances,
            "retirements": second.retirements,
            "remap_of_gone_topic": second.remap_document_topics([(1, 0.7)]),
        }

        # max_topics: three topics capped at two.
        config = AlignmentConfig(
            enabled=True, ledger_path=os.path.join(tmp, "cap.json"),
            similarity_threshold=0.30, min_shared_terms=2, topn=6, max_topics=2)
        capped = TopicAligner(config).align_and_commit(
            Model([[6, 6, 0, 0, 0, 0], [0, 0, 6, 6, 0, 0], [6, 5, 0.1, 0, 0, 0]]), "cap-1")
        cases["max_topics"] = {
            "current": capped.local_to_uid,
            "absorptions": capped.absorptions,
        }

        # Revival: a topic vanishes for one batch, then returns and reclaims its number.
        config = AlignmentConfig(
            enabled=True, ledger_path=os.path.join(tmp, "revive.json"),
            similarity_threshold=0.30, min_shared_terms=2, topn=6, retired_grace=2)
        reviver = TopicAligner(config)
        first = reviver.align_and_commit(
            Model([[6, 6, 0, 0, 0, 0], [0, 0, 6, 6, 0, 0]]), "revive-1")
        reviver.align_and_commit(Model([[6, 6, 0, 0, 0, 0]]), "revive-2")
        third = reviver.align_and_commit(
            Model([[0, 0, 6, 6, 0, 0], [6, 6, 0, 0, 0, 0]]), "revive-3")
        cases["revival"] = {
            "original_uid": first.local_to_uid[1],
            "revivals": third.revivals,
            "current": third.local_to_uid,
        }
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports", help="output directory")
    args = parser.parse_args()
    logging.getLogger("gensim").setLevel(logging.ERROR)

    os.makedirs(args.out, exist_ok=True)
    ledger_path = os.path.join(args.out, "topic-ledger.json")
    if os.path.exists(ledger_path):
        os.unlink(ledger_path)

    texts_a, texts_b = load_batches()
    dictionary = Dictionary(texts_a + texts_b)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus_a = [dictionary.doc2bow(text) for text in texts_a]
    corpus_b = [dictionary.doc2bow(text) for text in texts_b]

    config = AlignmentConfig(
        enabled=True, ledger_path=ledger_path, topn=50, min_shared_terms=5,
        similarity_threshold=0.30, drift_threshold=0.45)
    aligner = TopicAligner(config)

    model_a = train(corpus_a, dictionary, seed=11)
    first = aligner.align_and_commit(model_a, "2026-10")
    model_b = train(corpus_b, dictionary, seed=22)
    second = aligner.align_and_commit(model_b, "2026-11")

    mapping = []
    for match in sorted(second.matches, key=lambda m: m.new_local_index):
        mapping.append({
            "topic_uid": match.old.topic_uid,
            "previous_batch": "2026-10",
            "previous_local_index": match.old.local_index,
            "previous_top_words": topic_words(model_a, match.old.local_index),
            "current_batch": "2026-11",
            "current_local_index": match.new_local_index,
            "current_top_words": topic_words(model_b, match.new_local_index),
            "similarity": round(match.similarity, 4),
            "drift": None if match.drift is None else round(match.drift, 4),
            "drift_status": match.drift_status,
        })

    original = model_a.get_document_topics(corpus_a[0], minimum_probability=0.0)
    remapped, conservation = second.remap_document_topics(original)

    expected = {"batch_id": "2026-11", "local_to_uid": second.local_to_uid}
    cross_process = cross_process_check(ledger_path, expected)

    ledger = TopicLedger.read_only(ledger_path)
    report = {
        "generated_by": "reports/make_topic_alignment_report.py",
        "corpus": {
            "source": "gensim/test/test_data/lee_background.cor",
            "batch_a_documents": len(texts_a),
            "batch_b_documents": len(texts_b),
            "vocabulary": len(dictionary),
            "num_topics": NUM_TOPICS,
        },
        "config": config.to_dict(),
        "batch_2026_10": first.to_report(),
        "batch_2026_11": second.to_report(),
        "identifier_mapping": mapping,
        "document_remap_example": {
            "original": [[int(i), float(w)] for i, w in original],
            "remapped": [[int(i), float(w)] for i, w in remapped],
            "conservation": conservation,
        },
        "synthetic_fate_cases": synthetic_fate_cases(),
        "cross_process_read": cross_process,
        "ledger": {
            "path": ledger_path,
            "batches": ledger.batch_ids,
            "verify_problems": ledger.verify(),
            "fingerprint": ledger._payload.get("fingerprint"),
        },
    }

    json_path = os.path.join(args.out, "topic-alignment-report.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)

    md_path = os.path.join(args.out, "topic-alignment-report.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        write_markdown(handle, report)

    print(json.dumps({
        "json": json_path,
        "markdown": md_path,
        "matched": second.summary()["matched"],
        "born": second.summary()["born"],
        "merged": second.summary()["merged"],
        "split": second.summary()["split"],
        "mean_drift": second.summary()["mean_drift"],
        "cross_process_identical": cross_process["matches_in_process"],
        "ledger_verify_problems": ledger.verify(),
    }, indent=2))


def write_markdown(handle, report):
    handle.write("# Cross-batch stable topic numbering — acceptance report\n\n")
    handle.write("Generated by `reports/make_topic_alignment_report.py`. ")
    handle.write("This report contains topic identifiers, words and counts only; ")
    handle.write("no raw document content is included.\n\n")

    handle.write("## Setup\n\n")
    corpus = report["corpus"]
    handle.write("- Corpus: `%s`\n" % corpus["source"])
    handle.write("- Batch A: %d documents; Batch B: %d documents; vocabulary: %d\n"
                 % (corpus["batch_a_documents"], corpus["batch_b_documents"], corpus["vocabulary"]))
    handle.write("- Topics per batch: %d\n" % corpus["num_topics"])
    config = report["config"]
    handle.write("- similarity_threshold=%.2f, drift_threshold=%.2f, min_shared_terms=%d, topn=%d\n\n"
                 % (config["similarity_threshold"], config["drift_threshold"],
                    config["min_shared_terms"], config["topn"]))

    handle.write("## Identifier mapping (2026-10 -> 2026-11)\n\n")
    handle.write("| uid | previous # | previous top words | current # | "
                 "current top words | similarity | drift | status |\n")
    handle.write("|---|---|---|---|---|---|---|---|\n")
    for row in report["identifier_mapping"]:
        handle.write("| %s | %s | %s | %s | %s | %.3f | %s | %s |\n" % (
            row["topic_uid"], row["previous_local_index"],
            ", ".join(row["previous_top_words"]), row["current_local_index"],
            ", ".join(row["current_top_words"]), row["similarity"],
            "n/a" if row["drift"] is None else "%.3f" % row["drift"], row["drift_status"]))

    handle.write("\n## Batch summaries\n\n")
    for key in ("batch_2026_10", "batch_2026_11"):
        summary = report[key]["summary"]
        handle.write("**%s** — matched=%d, born=%d, merged=%d, split=%d, "
                     "disappeared=%d, retired=%d, absorbed=%d, mean_drift=%s\n\n" % (
                         summary["batch_id"], summary["matched"], summary["born"],
                         summary["merged"], summary["split"], summary["disappeared"],
                         summary["retired"], summary["absorbed"],
                         "n/a" if summary["mean_drift"] is None else "%.4f" % summary["mean_drift"]))

    handle.write("## Document distribution remap\n\n")
    example = report["document_remap_example"]
    handle.write("- original: `%s`\n" % example["original"])
    handle.write("- remapped: `%s`\n" % example["remapped"])
    handle.write("- input mass %.6f, output mass %.6f, dropped %.6f, conserved: `%s`\n\n" % (
        example["conservation"]["input_mass"], example["conservation"]["output_mass"],
        example["conservation"]["dropped_mass"], example["conservation"]["total_conserved"]))

    handle.write("## Synthetic fate cases (exact fixtures)\n\n")
    cases = report["synthetic_fate_cases"]
    merge = cases["merge"]
    handle.write("### Merge\n\n")
    handle.write("- two previous topics -> one current topic\n")
    handle.write("- merges: `%s`\n" % merge["merges"])
    handle.write("- remap of both previous topics: `%s` (conserved: %s)\n\n" % (
        merge["remap_of_both_previous_topics"][0],
        merge["remap_of_both_previous_topics"][1]["total_conserved"]))

    handle.write("### Split\n\n")
    for split in cases["split"]["splits"]:
        handle.write("- parent `%s` keeps local #%d (share %.3f); secondary `%s` gets local #%d (share %.3f)\n"
                     % (split["parent"], split["primary_new_local_index"], split["parent_share"],
                        split["secondary"], split["secondary_new_local_index"], split["secondary_share"]))
    handle.write("\n")

    handle.write("### Disappearance\n\n")
    for entry in cases["disappearance"]["disappearances"] + cases["disappearance"]["retirements"]:
        handle.write("- `%s` (%s): destination local #%s — %s\n"
                     % (entry["topic_uid"], entry["label"], entry["destination"], entry["reason"]))
    remap = cases["disappearance"]["remap_of_gone_topic"]
    handle.write("- remap of the vanished topic's documents: `%s`, dropped mass %.6f\n\n"
                 % (remap[0], remap[1]["dropped_mass"]))

    handle.write("### Revival\n\n")
    revival = cases["revival"]
    handle.write("- identifier `%s` vanished in one batch and reclaimed a topic in a later batch\n"
                 % revival["original_uid"])
    for entry in revival["revivals"]:
        handle.write("- `%s` -> local #%d (similarity %.3f): %s\n"
                     % (entry["topic_uid"], entry["new_local_index"],
                        entry["similarity"], entry["reason"]))
    handle.write("\n")

    handle.write("### max_topics\n\n")
    for absorption in cases["max_topics"]["absorptions"]:
        handle.write("- `%s` absorbed into `%s` (similarity %.3f) — %s\n"
                     % (absorption["absorbed"], absorption["into"],
                        absorption["similarity"], absorption["reason"]))
    handle.write("\n")

    handle.write("## Cross-process ledger read\n\n")
    cross = report["cross_process_read"]
    handle.write("- batches read by a separate interpreter: `%s`\n" % cross["batches"])
    handle.write("- identifiers identical to the writing process: `%s`\n"
                 % cross["matches_in_process"])
    handle.write("- ledger internal checks: `%s`\n" % (cross["verify"] or "no problems"))
    handle.write("- ledger fingerprint: `%s`\n" % report["ledger"]["fingerprint"])


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    main()
