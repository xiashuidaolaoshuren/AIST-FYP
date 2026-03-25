"""Compatibility wrapper for the renamed oracle-track converter.

Deprecated: use `scripts/convert_citebench_oracle_to_lettucedetect.py`.
"""

from __future__ import annotations

from scripts.convert_citebench_oracle_to_lettucedetect import main


if __name__ == "__main__":
    raise SystemExit(main())
