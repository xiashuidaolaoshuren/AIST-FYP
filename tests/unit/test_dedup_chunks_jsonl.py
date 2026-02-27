import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            if isinstance(row, str):
                f.write(row + '\n')
            else:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _read_jsonl(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def test_dedup_output_file(tmp_path):
    input_path = tmp_path / 'chunks.jsonl'
    output_path = tmp_path / 'chunks_dedup.jsonl'

    rows = [
        {'doc_id': 'a', 'sent_id': 0, 'text': 'x'},
        {'doc_id': 'a', 'sent_id': 0, 'text': 'x'},
        {'doc_id': 'a', 'sent_id': 1, 'text': 'y'},
        {'doc_id': 'b', 'sent_id': 0, 'text': 'z'},
    ]
    _write_jsonl(input_path, rows)

    cmd = [
        sys.executable,
        'scripts/dedup_chunks_jsonl.py',
        '--input', str(input_path),
        '--output', str(output_path),
        '--backend', 'memory',
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).parents[2], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)

    assert report['total_lines'] == 4
    assert report['kept_lines'] == 3
    assert report['duplicate_lines'] == 1
    dedup_rows = _read_jsonl(output_path)
    assert len(dedup_rows) == 3
    assert dedup_rows[0]['sent_id'] == 0
    assert dedup_rows[1]['sent_id'] == 1


def test_dedup_in_place_with_backup(tmp_path):
    input_path = tmp_path / 'chunks.jsonl'
    rows = [
        {'doc_id': 'a', 'sent_id': 0, 'text': 'x'},
        {'doc_id': 'a', 'sent_id': 0, 'text': 'x'},
    ]
    _write_jsonl(input_path, rows)

    cmd = [
        sys.executable,
        'scripts/dedup_chunks_jsonl.py',
        '--input', str(input_path),
        '--in-place',
        '--backend', 'memory',
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).parents[2], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)

    backup_path = Path(report['backup_path'])
    assert input_path.exists()
    assert backup_path.exists()
    dedup_rows = _read_jsonl(input_path)
    assert len(dedup_rows) == 1


def test_dedup_invalid_json_lines(tmp_path):
    input_path = tmp_path / 'chunks.jsonl'
    output_path = tmp_path / 'chunks_dedup.jsonl'

    rows = [
        {'doc_id': 'a', 'sent_id': 0, 'text': 'x'},
        '{not_json}',
        {'doc_id': 'a', 'sent_id': 0, 'text': 'x'},
    ]
    _write_jsonl(input_path, rows)

    cmd = [
        sys.executable,
        'scripts/dedup_chunks_jsonl.py',
        '--input', str(input_path),
        '--output', str(output_path),
        '--backend', 'memory',
        '--keep-invalid-lines',
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).parents[2], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)

    assert report['invalid_json_lines'] == 1
    assert report['kept_lines'] == 2

    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    assert len(lines) == 2
