#!/usr/bin/env python3
"""Minimal runner for the pytest-style test functions in tests/ (pytest is not
installed in the vjepa2-312 environment)."""

from __future__ import annotations

import importlib
import inspect
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def modules():
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        rel = path.relative_to(ROOT).with_suffix("")
        yield ".".join(rel.parts), path


def main() -> int:
    passed = failed = 0
    failures = []
    for name, path in modules():
        source = path.read_text(encoding="utf-8")
        if "unittest.TestCase" in source:
            continue  # covered by `python -m unittest discover`
        try:
            module = importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001
            failed += 1
            failures.append((name, "import", f"{type(exc).__name__}: {exc}"))
            continue
        for fname, fn in sorted(vars(module).items()):
            if not fname.startswith("test_") or not callable(fn):
                continue
            if inspect.signature(fn).parameters:
                continue  # needs fixtures; pytest-only
            try:
                fn()
                passed += 1
            except BaseException as exc:  # noqa: BLE001
                failed += 1
                failures.append((name, fname, f"{type(exc).__name__}: {exc}"))

    print(f"pytest-style functions passed: {passed}")
    print(f"pytest-style failures: {len(failures)}")
    for mod, fn, err in failures:
        print(f"  {mod}::{fn}\n      {err[:300]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
