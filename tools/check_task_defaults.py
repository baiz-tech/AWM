#!/usr/bin/env python3
"""Check that every task's config defaults are accepted by its module.

For each ``tasks.<task>`` with a ``launch.module`` in every config, run the
module with ``--help`` under a synthetic launcher environment. ``--help`` exits
before any heavy import or data access, but ``apply_cli_defaults`` runs first, so
this exercises the whole contract:

* ``CONFIG`` / ``TASK`` are read and the config block is found;
* every ``cli_defaults`` option exists in the module's parser;
* every template (``{experiment_root}``, ``{resolved_config}``, ...) renders.

A non-zero exit means the task would fail at startup.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def tasks():
    for config in sorted((ROOT / "configs").rglob("config.yaml")):
        spec = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        for name, entry in (spec.get("tasks") or {}).items():
            module = ((entry or {}).get("launch") or {}).get("module")
            if module:
                yield config, name, module


def main() -> int:
    failures = []
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for config, task, module in tasks():
            total += 1
            task_dir = Path(tmp) / task
            (task_dir / "configs").mkdir(parents=True, exist_ok=True)
            # The launcher writes the task-resolved config before starting the
            # module, and a few modules read it eagerly at parse time.
            spec = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
            block = (((spec.get("tasks") or {}).get(task) or {}).get("experiment")) or {}
            (task_dir / "configs" / f"{task}.resolved.yaml").write_text(
                yaml.safe_dump({**block, "folder": str(task_dir)}, sort_keys=False),
                encoding="utf-8",
            )
            env = dict(os.environ)
            env.update({
                "CONFIG": str(config),
                "TASK": task,
                "LARGE_DATA_OUTPUT_DIR": str(task_dir),
                "LIGHT_DATA_OUTPUT_DIR": str(task_dir),
                "RESUME": "0",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
            })
            result = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                cwd=ROOT, env=env, capture_output=True, text=True,
            )
            if result.returncode != 0:
                tail = (result.stderr or result.stdout).strip().split("\n")
                failures.append((f"{config.relative_to(ROOT)}:{task}", module,
                                 " | ".join(tail[-3:])[:300]))

    print(f"tasks checked: {total}")
    print(f"failures: {len(failures)}")
    for where, module, message in failures:
        print(f"\n  {where}\n    module: {module}\n    {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
