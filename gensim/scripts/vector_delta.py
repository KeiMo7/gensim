#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Gensim Contributors
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Command line front-end for :mod:`gensim.models.vector_delta`.

Compare two word-vector versions, inspect a single key, or apply a
delta-only incremental update.

Examples
--------

.. sourcecode:: bash

    # Compare two versions, write JSON + Markdown reports
    python -m gensim.scripts.vector_delta compare old.kv new.kv \\
        --report-json diff.json --report-markdown diff.md

    # Force the streaming (disk-spilling) mode for a million-key vocabulary
    python -m gensim.scripts.vector_delta compare old.kv new.kv --streaming

    # Explain one key
    python -m gensim.scripts.vector_delta explain old.kv new.kv --key "New York"

    # Apply a delta and keep the previous version for rollback
    python -m gensim.scripts.vector_delta update old.kv new.kv updated.kv \\
        --backup-dir updated.kv.prev --record update.json --overwrite

    # Roll back an update
    python -m gensim.scripts.vector_delta restore updated.kv --backup-dir updated.kv.prev

Exit codes: 0 success, 2 refused (dimension mismatch / missing base / unsupported
model), 1 other errors.

Command line arguments
----------------------

.. program-output:: python -m gensim.scripts.vector_delta --help
   :ellipsis: 0, -5

"""
import argparse
import json
import logging
import sys

from gensim.models import vector_delta


logger = logging.getLogger(__name__)


def _add_normalizer_args(parser):
    group = parser.add_argument_group('key identity (normalization)')
    group.add_argument(
        '--case-insensitive', action='store_true',
        help='compare keys after Unicode case folding (default: case-sensitive)',
    )
    group.add_argument(
        '--keep-whitespace', action='store_true',
        help='do not strip/collapse whitespace in keys (default: strip and collapse)',
    )
    group.add_argument(
        '--unicode-normalization', default='NFC', choices=['NFC', 'NFD', 'NFKC', 'NFKD', 'none'],
        help='Unicode normalization form applied to keys (default: NFC; "none" disables it)',
    )


def _normalizer(args):
    return vector_delta.KeyNormalizer(
        case_sensitive=not args.case_insensitive,
        strip=not args.keep_whitespace,
        collapse_whitespace=not args.keep_whitespace,
        unicode_normalization=None if args.unicode_normalization == 'none' else args.unicode_normalization,
    )


def _add_policy_args(parser):
    group = parser.add_argument_group('change thresholds')
    group.add_argument(
        '--relative-l2-threshold', type=float, default=vector_delta.ChangePolicy.relative_l2_threshold,
        help='relative L2 delta above which a key counts as changed (default: %(default)s)',
    )
    group.add_argument(
        '--cosine-distance-threshold', type=float, default=vector_delta.ChangePolicy.cosine_distance_threshold,
        help='cosine distance above which a key counts as changed (default: %(default)s)',
    )


def _policy(args):
    return vector_delta.ChangePolicy(
        relative_l2_threshold=args.relative_l2_threshold,
        cosine_distance_threshold=args.cosine_distance_threshold,
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m gensim.scripts.vector_delta',
        description=__doc__.split('Examples')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--verbose', action='store_true', help='enable INFO logging')
    subparsers = parser.add_subparsers(dest='command', required=True)

    compare = subparsers.add_parser('compare', help='compare two model versions')
    compare.add_argument('old', help='previous version (.kv/.model or word2vec format)')
    compare.add_argument('new', help='new version (.kv/.model or word2vec format)')
    compare.add_argument(
        '--format', default='auto', choices=['auto', 'kv', 'word2vec-text', 'word2vec-binary'],
        help='input format for both files (default: auto)',
    )
    compare.add_argument(
        '--streaming', dest='streaming', action='store_true', default=None,
        help='force disk-spilling streaming mode (default: on for file paths)',
    )
    compare.add_argument(
        '--no-streaming', dest='streaming', action='store_false',
        help='force in-memory mode (requires both models to fit in RAM)',
    )
    compare.add_argument('--chunk-size', type=int, default=10_000, help='vectors per streamed chunk')
    compare.add_argument('--max-examples', type=int, default=20, help='example keys embedded in the report')
    compare.add_argument(
        '--neighbor-examples', type=int, default=0,
        help='compute old/new nearest neighbours for this many changed keys (loads both models)',
    )
    compare.add_argument('--neighbor-topn', type=int, default=10, help='neighbour list length (default: 10)')
    compare.add_argument('--pairs', help='word-pair similarity file (e.g. wordsim353.tsv) for before/after metrics')
    compare.add_argument('--analogies', help='analogy file (e.g. questions-words.txt) for before/after metrics')
    compare.add_argument('--report-json', help='write the JSON report here')
    compare.add_argument('--report-markdown', help='write the Markdown report here')
    _add_normalizer_args(compare)
    _add_policy_args(compare)

    explain = subparsers.add_parser('explain', help='explain one key between two versions')
    explain.add_argument('old', help='previous version')
    explain.add_argument('new', help='new version')
    explain.add_argument('--key', required=True, help='key to look up (matched under the normalization rules)')
    explain.add_argument('--topn', type=int, default=10, help='nearest neighbours to report (default: 10)')
    explain.add_argument(
        '--format', default='auto', choices=['auto', 'kv', 'word2vec-text', 'word2vec-binary'],
        help='input format for both files (default: auto)',
    )
    explain.add_argument('--report-json', help='write the key delta as JSON here (default: stdout)')
    _add_normalizer_args(explain)
    _add_policy_args(explain)

    update = subparsers.add_parser('update', help='apply a delta-only incremental update')
    update.add_argument('base', help='previous version to build on')
    update.add_argument('updates', help='delta source (only changed/new vectors)')
    update.add_argument('output', help='path for the updated model')
    update.add_argument(
        '--format', default='auto', choices=['auto', 'kv', 'word2vec-text', 'word2vec-binary'],
        help='input format for base/updates when they are word2vec files',
    )
    update.add_argument('--prune-missing', action='store_true',
                        help='drop base keys absent from the delta (default: keep them)')
    update.add_argument('--overwrite', action='store_true', help='allow replacing an existing output')
    update.add_argument('--no-backup', dest='backup', action='store_false',
                        help='do not keep the previous output for rollback')
    update.add_argument('--backup-dir', help='directory for the previous output (default: <output>.prev)')
    update.add_argument('--record', help='write the update record JSON here')
    _add_normalizer_args(update)
    _add_policy_args(update)

    restore = subparsers.add_parser('restore', help='roll back to a backup produced by update')
    restore.add_argument('output', help='the updated model path to roll back')
    restore.add_argument('--backup-dir', required=True, help='backup directory from the update record')
    restore.add_argument('--remove-backup', action='store_true', help='delete the backup after restoring')
    return parser


def _run_compare(args):
    similarity_tasks = None
    if args.pairs or args.analogies:
        similarity_tasks = {'pairs': args.pairs, 'analogies': args.analogies}
    report = vector_delta.compare_versions(
        args.old, args.new,
        normalizer=_normalizer(args), policy=_policy(args),
        streaming=args.streaming, format=args.format, chunk_size=args.chunk_size,
        max_examples=args.max_examples,
        nearest_neighbor_examples=args.neighbor_examples, nearest_neighbor_topn=args.neighbor_topn,
        similarity_tasks=similarity_tasks,
    )
    data = report.to_dict()
    vocab = data['vocabulary']
    print(
        "status=%(status)s old_keys=%(ok)d new_keys=%(nk)d added=%(a)d removed=%(r)d "
        "common=%(c)d changed=%(ch)d unchanged_exact=%(ex)d unchanged_within_threshold=%(wt)d"
        % {
            'status': data['status'], 'ok': vocab['old_keys'], 'nk': vocab['new_keys'],
            'a': vocab['added'], 'r': vocab['removed'], 'c': vocab['common'],
            'ch': vocab['changed'], 'ex': vocab['unchanged_exact'],
            'wt': vocab['unchanged_within_threshold'],
        }
    )
    if args.report_json:
        report.write_json(args.report_json)
        print(f'wrote {args.report_json}')
    if args.report_markdown:
        report.write_markdown(args.report_markdown)
        print(f'wrote {args.report_markdown}')
    return 0


def _run_explain(args):
    delta = vector_delta.explain_key(
        args.old, args.new, args.key,
        normalizer=_normalizer(args), policy=_policy(args), topn=args.topn,
    )
    payload = json.dumps(delta.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
    if args.report_json:
        with open(args.report_json, 'w', encoding='utf8') as fout:
            fout.write(payload + '\n')
        print(f'wrote {args.report_json}')
    else:
        print(payload)
    return 0


def _run_update(args):
    record = vector_delta.incremental_update(
        args.base, args.updates, args.output,
        normalizer=_normalizer(args), policy=_policy(args), format=args.format,
        prune_missing=args.prune_missing, overwrite=args.overwrite,
        backup_dir=args.backup_dir, backup=args.backup,
    )
    data = record.to_dict()
    print(
        'output=%(out)s rows_total=%(total)d reused_exact=%(reuse)d replaced=%(repl)d added=%(add)d '
        'unchanged_kept=%(keep)d pruned=%(prune)d backup=%(backup)s elapsed_s=%(elapsed).3f'
        % {
            'out': data['output_path'], 'total': data['rows_total'],
            'reuse': data['rows_reused_exact'], 'repl': data['rows_replaced'],
            'add': data['rows_added'], 'keep': data['rows_unchanged_kept'],
            'prune': data['rows_pruned'], 'backup': data['backup_dir'],
            'elapsed': data['elapsed_seconds'],
        }
    )
    if args.record:
        record.write_json(args.record)
        print(f'wrote {args.record}')
    return 0


def _run_restore(args):
    restored = vector_delta.restore_backup(
        args.output, args.backup_dir, remove_backup=args.remove_backup,
    )
    print(f'restored {len(restored)} file(s) to {args.output}')
    return 0


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
        level=logging.INFO if args.verbose else logging.WARNING,
    )
    handlers = {
        'compare': _run_compare,
        'explain': _run_explain,
        'update': _run_update,
        'restore': _run_restore,
    }
    try:
        return handlers[args.command](args)
    except (
        vector_delta.DimensionMismatchError,
        vector_delta.BaseVersionMissingError,
        vector_delta.UnsupportedModelError,
    ) as err:
        print(f'refused: {err}', file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError, FileExistsError) as err:
        print(f'error: {err}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
