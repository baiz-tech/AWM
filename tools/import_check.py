#!/usr/bin/env python3
"""Import every migrated module and classify failures.

Run from the makeup/ root so that `src.*` and `external.*` resolve.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKIP_PARTS = {"legacy", "__pycache__", "configs_legacy"}


def module_name(path: Path) -> str | None:
    rel = path.relative_to(ROOT)
    if SKIP_PARTS & set(rel.parts):
        return None
    if path.name == "__init__.py":
        return None
    return ".".join(rel.with_suffix("").parts)


def main() -> None:
    ok, failed = 0, []
    for path in sorted((ROOT / "src").rglob("*.py")):
        name = module_name(path)
        if name is None:
            continue
        try:
            importlib.import_module(name)
            ok += 1
        except BaseException as exc:  # noqa: BLE001 - report everything
            failed.append((name, f"{type(exc).__name__}: {exc}"))

    print(f"imported OK: {ok}")
    print(f"failed: {len(failed)}\n")
    for name, err in failed:
        print(f"  {name}\n      {err[:220]}")


if __name__ == "__main__":
    main()
