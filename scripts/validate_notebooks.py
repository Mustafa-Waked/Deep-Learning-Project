#!/usr/bin/env python3
"""Smoke-check notebook JSON and optional dependency imports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["DataCleaning.ipynb", "ProjectCode.ipynb"]


def main() -> int:
    errors = 0
    for name in NOTEBOOKS:
        path = ROOT / name
        if not path.is_file():
            print(f"MISSING: {name}")
            errors += 1
            continue
        try:
            json.load(path.open(encoding="utf-8"))
            print(f"OK: {name} (valid JSON)")
        except json.JSONDecodeError as exc:
            print(f"FAIL: {name} — {exc}")
            errors += 1

    py = sys.version_info
    print(f"Python {py.major}.{py.minor}.{py.micro}")
    if py >= (3, 13):
        print(
            "WARN: TensorFlow may not install on Python 3.13+. "
            "Use Python 3.9–3.12 for full notebook execution."
        )

    try:
        import tensorflow as tf  # noqa: F401

        print(f"OK: tensorflow {tf.__version__}")
    except ImportError:
        print("SKIP: tensorflow not installed (expected if Python version is unsupported)")

    return errors


if __name__ == "__main__":
    raise SystemExit(main())
