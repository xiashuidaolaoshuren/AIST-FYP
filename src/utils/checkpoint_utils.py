"""
Shared checkpoint helpers for long-running data processing scripts.

Provides lightweight manifest save/load, strict compatibility checks,
and file fingerprint utilities used by chunking and index-building scripts.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable


CHECKPOINT_SCHEMA_VERSION = 1


def file_fingerprint(path: Path) -> Dict[str, Any]:
    """Create a lightweight fingerprint for compatibility checks."""
    stat = path.stat()
    return {
        'path': str(path),
        'size_bytes': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def save_manifest(manifest_path: Path, data: Dict[str, Any]) -> None:
    """Atomically save a JSON checkpoint manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + '.tmp')

    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    tmp_path.replace(manifest_path)


def write_pickle_atomic(file_path: Path, data: Any) -> None:
    """Atomically write pickle data with fsync durability."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + '.tmp')

    with open(tmp_path, 'wb') as f:
        pickle.dump(data, f)
        f.flush()
        os.fsync(f.fileno())

    tmp_path.replace(file_path)


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load a JSON checkpoint manifest."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_manifest_compatible(
    manifest: Dict[str, Any],
    expected: Dict[str, Any],
    required_keys: Iterable[str],
) -> None:
    """Raise ValueError if required fields are missing or expected fields mismatch."""
    for key in required_keys:
        if key not in manifest:
            raise ValueError(f"Checkpoint missing required key: {key}")

        if key in expected and manifest[key] != expected[key]:
            raise ValueError(
                f"Checkpoint mismatch for '{key}': "
                f"checkpoint={manifest[key]!r}, expected={expected[key]!r}"
            )


def truncate_to_last_complete_jsonl_line(file_path: Path) -> int:
    """
    Truncate file to the last newline boundary.

    Returns:
        New file size in bytes after truncation.
    """
    if not file_path.exists():
        return 0

    with open(file_path, 'rb+') as f:
        f.seek(0, 2)
        file_size = f.tell()

        if file_size == 0:
            return 0

        f.seek(file_size - 1)
        if f.read(1) == b'\n':
            return file_size

        pos = file_size - 1
        while pos >= 0:
            f.seek(pos)
            if f.read(1) == b'\n':
                truncate_size = pos + 1
                f.truncate(truncate_size)
                return truncate_size
            pos -= 1

        f.truncate(0)
        return 0
