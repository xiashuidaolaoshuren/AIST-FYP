"""
Deduplicate chunk JSONL files by key fields.

Usage examples:
    python scripts/dedup_chunks_jsonl.py --input data/processed/wiki_chunks_production.jsonl --in-place
    python scripts/dedup_chunks_jsonl.py --input in.jsonl --output out.jsonl --backend memory
    python scripts/dedup_chunks_jsonl.py --input in.jsonl --output out.jsonl --no-progress
"""

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from tqdm import tqdm


@dataclass
class DedupStats:
    total_lines: int = 0
    kept_lines: int = 0
    duplicate_lines: int = 0
    invalid_json_lines: int = 0

    def to_dict(self) -> dict:
        return {
            'total_lines': self.total_lines,
            'kept_lines': self.kept_lines,
            'duplicate_lines': self.duplicate_lines,
            'invalid_json_lines': self.invalid_json_lines,
        }


class InMemoryKeyStore:
    def __init__(self) -> None:
        self._keys = set()

    def add_if_absent(self, key: str) -> bool:
        if key in self._keys:
            return False
        self._keys.add(key)
        return True

    def close(self) -> None:
        return None


class SQLiteKeyStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute('PRAGMA journal_mode = WAL')
        self.conn.execute('PRAGMA synchronous = NORMAL')
        self.conn.execute('PRAGMA temp_store = MEMORY')
        self.conn.execute('CREATE TABLE IF NOT EXISTS keys_seen (k TEXT PRIMARY KEY)')
        self.conn.commit()

    def add_if_absent(self, key: str) -> bool:
        cursor = self.conn.execute('INSERT OR IGNORE INTO keys_seen(k) VALUES (?)', (key,))
        return cursor.rowcount == 1

    def close(self) -> None:
        try:
            self.conn.close()
        finally:
            if self.db_path.exists():
                self.db_path.unlink()


def parse_key_fields(value: str) -> List[str]:
    fields = [item.strip() for item in value.split(',') if item.strip()]
    if not fields:
        raise argparse.ArgumentTypeError('key-fields must contain at least one field name')
    return fields


def build_key(record: dict, key_fields: Iterable[str]) -> str:
    field_values = []
    missing = False
    for field in key_fields:
        if field not in record:
            missing = True
            break
        field_values.append(record[field])

    if missing:
        return 'raw:' + json.dumps(record, sort_keys=True, ensure_ascii=False)

    return 'fields:' + json.dumps(field_values, ensure_ascii=False, separators=(',', ':'))


def deduplicate_jsonl(
    input_path: Path,
    output_path: Path,
    key_fields: List[str],
    backend: str = 'sqlite',
    keep_invalid_lines: bool = False,
    show_progress: bool = True,
    progress_refresh_lines: int = 5000,
) -> DedupStats:
    if backend not in {'sqlite', 'memory'}:
        raise ValueError(f'Unsupported backend: {backend}')

    if backend == 'memory':
        key_store = InMemoryKeyStore()
    else:
        db_path = output_path.with_suffix(output_path.suffix + '.keys.sqlite')
        key_store = SQLiteKeyStore(db_path)

    stats = DedupStats()
    input_size_bytes = input_path.stat().st_size
    progress_bar = None

    try:
        if show_progress:
            progress_bar = tqdm(
                total=input_size_bytes,
                unit='B',
                unit_scale=True,
                desc='Deduplicating chunks',
            )

        with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                stats.total_lines += 1
                if progress_bar:
                    line_size_bytes = len(line.encode('utf-8'))
                    progress_bar.update(line_size_bytes)

                if progress_bar and (stats.total_lines % progress_refresh_lines == 0):
                    progress_bar.set_postfix(
                        kept=stats.kept_lines,
                        dups=stats.duplicate_lines,
                        invalid=stats.invalid_json_lines,
                    )

                stripped = line.strip()
                if not stripped:
                    stats.invalid_json_lines += 1
                    if keep_invalid_lines:
                        fout.write(line)
                        stats.kept_lines += 1
                    continue

                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    stats.invalid_json_lines += 1
                    if keep_invalid_lines:
                        fout.write(line)
                        stats.kept_lines += 1
                    continue

                key = build_key(record, key_fields)
                if key_store.add_if_absent(key):
                    fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stats.kept_lines += 1
                else:
                    stats.duplicate_lines += 1

            if progress_bar:
                progress_bar.set_postfix(
                    kept=stats.kept_lines,
                    dups=stats.duplicate_lines,
                    invalid=stats.invalid_json_lines,
                )
    finally:
        if progress_bar:
            progress_bar.close()
        key_store.close()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description='Deduplicate chunk JSONL by key fields')
    parser.add_argument('--input', required=True, help='Input JSONL path')
    parser.add_argument('--output', default=None, help='Output JSONL path (required unless --in-place)')
    parser.add_argument('--in-place', action='store_true', help='Replace input file atomically with deduplicated output')
    parser.add_argument(
        '--key-fields',
        type=parse_key_fields,
        default=['doc_id', 'sent_id'],
        help='Comma-separated key fields for duplicate detection (default: doc_id,sent_id)',
    )
    parser.add_argument('--backend', choices=['sqlite', 'memory'], default='sqlite', help='Key store backend')
    parser.add_argument('--keep-invalid-lines', action='store_true', help='Keep malformed JSON lines in output')
    parser.add_argument('--report-json', default=None, help='Optional path to write dedup report JSON')
    parser.add_argument('--no-backup', action='store_true', help='Disable .bak backup when using --in-place')
    parser.add_argument('--no-progress', action='store_true', help='Disable tqdm progress display')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    if args.in_place:
        output_path = input_path.with_suffix(input_path.suffix + '.dedup.tmp')
    else:
        if not args.output:
            raise ValueError('--output is required when not using --in-place')
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = deduplicate_jsonl(
        input_path=input_path,
        output_path=output_path,
        key_fields=args.key_fields,
        backend=args.backend,
        keep_invalid_lines=args.keep_invalid_lines,
        show_progress=not args.no_progress,
    )

    report = {
        'input_path': str(input_path),
        'output_path': str(input_path if args.in_place else output_path),
        'key_fields': args.key_fields,
        'backend': args.backend,
        **stats.to_dict(),
    }

    if args.in_place:
        backup_path: Optional[Path] = None
        if not args.no_backup:
            backup_path = input_path.with_suffix(input_path.suffix + '.bak')
            if backup_path.exists():
                backup_path.unlink()
            input_path.replace(backup_path)
            output_path.replace(input_path)
            report['backup_path'] = str(backup_path)
        else:
            output_path.replace(input_path)
            report['backup_path'] = None

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
