#!/usr/bin/env python3
"""Run the CLEVRER QA probe pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from src.core.config_utils import merged_task_experiment
from src.data.clevrer.v1.protocol import resolve_qa_protocol


def _command_text(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def _torchrun(python_bin, master_addr, port, num_gpus, module, args):
    return [
        python_bin,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr",
        master_addr,
        "--master_port",
        str(port),
        "--nproc_per_node",
        str(num_gpus),
        "-m",
        module,
        *args,
    ]


def _archive_path(path):
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = path.parent / "legacy" / f"{path.name}_{timestamp}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while archive.exists():
        archive = path.parent / "legacy" / f"{path.name}_{timestamp}_{suffix}"
        suffix += 1
    shutil.move(str(path), str(archive))
    print(f"Archived existing path: {path} -> {archive}", flush=True)
    return archive


def _archive_existing_paths(paths):
    seen = set()
    archived = []
    for path in paths:
        path = Path(path)
        key = path.resolve() if path.exists() else path.absolute()
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            archived.append(_archive_path(path))
    return archived


def _mirror_lightweight_outputs(source, destination):
    """Mirror metadata without duplicating trajectories or checkpoints."""
    suffixes = {".json", ".log", ".txt", ".yaml", ".yml", ".exitcode", ".pid"}
    source = Path(source)
    destination = Path(destination)
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _load_qa_config(config_path):
    with open(config_path, encoding="utf-8") as handle:
        config = merged_task_experiment(yaml.safe_load(handle), "qa_eval")
    return config.get("qa", {}) or {}


def _validate_protocol(config_path):
    with open(config_path, encoding="utf-8") as handle:
        config = merged_task_experiment(yaml.safe_load(handle), "qa_eval")
    return resolve_qa_protocol(config)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _trajectory_export_complete(split_dir, min_scenes=None):
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    trajectories = list(split_dir.glob("scene_*.pt"))
    if not trajectories:
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    manifest_scenes = manifest.get("scenes")
    if (
        isinstance(manifest_scenes, int)
        and manifest_scenes > 0
        and len(trajectories) < manifest_scenes
    ):
        return False
    if min_scenes is not None and len(trajectories) < int(min_scenes):
        return False
    return True


def _write_skip_stage(name, split_dir, primary_log, mirror_log=None):
    message = (
        f"{name} skipped: existing QA trajectories detected under {split_dir}\n"
    )
    print(message, end="", flush=True)
    primary_log.parent.mkdir(parents=True, exist_ok=True)
    primary_log.write_text(message, encoding="utf-8")
    if mirror_log is not None:
        mirror_log.parent.mkdir(parents=True, exist_ok=True)
        mirror_log.write_text(message, encoding="utf-8")


def _run_stage(name, command, primary_log, mirror_log=None):
    primary_log.parent.mkdir(parents=True, exist_ok=True)
    if mirror_log is not None:
        mirror_log.parent.mkdir(parents=True, exist_ok=True)
        mirror_log.write_text("", encoding="utf-8")
    print(f"{name} command: {_command_text(command)}", flush=True)
    with primary_log.open("w", encoding="utf-8") as primary_handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            primary_handle.write(line)
            primary_handle.flush()
            if mirror_log is not None:
                with mirror_log.open("a", encoding="utf-8") as mirror_handle:
                    mirror_handle.write(line)
        status = process.wait()
    if status != 0:
        raise subprocess.CalledProcessError(status, command)


def _run_export_stage(
    name,
    command,
    trajectory_root,
    split,
    min_scenes,
    skip_existing,
    primary_log,
    mirror_log=None,
):
    split_dir = trajectory_root / split
    if skip_existing and _trajectory_export_complete(split_dir, min_scenes=min_scenes):
        _write_skip_stage(name, split_dir, primary_log, mirror_log)
        return
    _run_stage(name, command, primary_log, mirror_log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace-run-dir", required=True)
    parser.add_argument("--data-run-dir", required=True)
    parser.add_argument("--workspace-output-dir", required=True)
    parser.add_argument("--data-output-dir", required=True)
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--base-port", required=True, type=int)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--num-gpus", required=True, type=int)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--output-mode",
        choices=("both", "data", "workspace"),
        default=os.environ.get("OUTPUT_MODE", "workspace"),
    )
    parser.add_argument("--variant", default=os.environ.get("VARIANT", "naive"))
    parser.add_argument("--archive-existing", action="store_true")
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-val-scenes", type=int)
    parser.add_argument("--max-eval-scenes", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    if args.variant not in {"naive", "observed_only", "copy_future"}:
        raise ValueError(f"unsupported variant: {args.variant}")

    cfg_qa = _load_qa_config(args.config)
    protocol_info = _validate_protocol(args.config)
    if protocol_info["token_source"] != "online":
        raise ValueError(
            "qa_pipeline currently exports online trajectories; precomputed trajectories "
            "should invoke train_qa/eval_qa directly"
        )
    skip_existing_trajectories = _as_bool(
        cfg_qa.get("skip_existing_trajectories", False)
    )
    archive_existing_probe_outputs = _as_bool(
        cfg_qa.get("archive_existing_probe_outputs", True)
    )

    workspace_run_dir = Path(args.workspace_run_dir)
    data_run_dir = Path(args.data_run_dir)
    workspace_qa_root = Path(args.workspace_output_dir)
    data_qa_root = Path(args.data_output_dir)
    world_checkpoint = Path(args.world_checkpoint)
    primary_qa_root = data_qa_root if args.output_mode in {"both", "data"} else workspace_qa_root
    mirror_qa_root = workspace_qa_root if args.output_mode == "both" else None
    trajectory_base = primary_qa_root / "qa_trajectories"
    trajectory_root = trajectory_base / args.variant
    qa_dir = primary_qa_root / "qa_checkpoints" / args.variant
    qa_eval_dir = primary_qa_root / "validation"
    qa_test_dir = primary_qa_root / "test"
    primary_status_file = primary_qa_root / f"qa_{args.variant}_pipeline.exitcode"
    mirror_status_file = (
        mirror_qa_root / f"qa_{args.variant}_pipeline.exitcode"
        if mirror_qa_root is not None else None
    )
    primary_stage_logs = [
        primary_qa_root / f"qa_{args.variant}_{name}.log"
        for name in (
            "export_train",
            "export_validation",
            "export_test",
            "train",
            "eval_validation",
            "eval_test",
        )
    ]
    mirror_stage_logs = (
        [mirror_qa_root / f"qa_{args.variant}_{name}.log" for name in (
            "export_train", "export_validation", "export_test", "train",
            "eval_validation", "eval_test",
        )] if mirror_qa_root is not None else []
    )
    primary_run_dir = data_run_dir if args.output_mode in {"both", "data"} else workspace_run_dir
    legacy_root_pipeline_files = [
        primary_run_dir / f"qa_{args.variant}_pipeline.exitcode",
        primary_run_dir / f"qa_{args.variant}_pipeline.log",
        primary_run_dir / f"qa_{args.variant}_pipeline.pid",
        *[
            primary_run_dir / f"qa_{args.variant}_{name}.log"
            for name in (
                "export_train",
                "export_validation",
                "export_test",
                "train",
                "eval_validation",
                "eval_test",
            )
        ],
    ]

    if not world_checkpoint.is_file():
        raise FileNotFoundError(f"World-model checkpoint not found: {world_checkpoint}")

    existing = [qa_dir, qa_eval_dir, qa_test_dir]
    if mirror_qa_root is not None:
        existing.extend([mirror_qa_root / "validation", mirror_qa_root / "test"])
    present = [path for path in existing if path.exists()]
    should_archive = args.archive_existing or archive_existing_probe_outputs
    if present:
        if not should_archive:
            raise FileExistsError(
                "QA output already exists; enable qa.archive_existing_probe_outputs "
                "or use --archive-existing to start a fresh probe run"
            )
        _archive_existing_paths(present)

    if should_archive:
        _archive_existing_paths(
            [
                primary_status_file,
                *([] if mirror_status_file is None else [mirror_status_file]),
                *primary_stage_logs,
                *mirror_stage_logs,
                *legacy_root_pipeline_files,
            ]
        )

    primary_qa_root.mkdir(parents=True, exist_ok=True)
    if mirror_qa_root is not None:
        mirror_qa_root.mkdir(parents=True, exist_ok=True)
    primary_status_file.write_text("", encoding="utf-8")
    if mirror_status_file is not None:
        mirror_status_file.write_text("", encoding="utf-8")

    val_export_scenes = args.max_eval_scenes
    if args.max_val_scenes is not None:
        val_export_scenes = max(val_export_scenes or 0, args.max_val_scenes)

    def optional(flag, value):
        return [] if value is None else [flag, str(value)]

    export_train = _torchrun(
        args.python_bin,
        args.master_addr,
        args.base_port + 0,
        args.num_gpus,
        "recipe.shared.evaluate.clevrer.export_qa_trajectories",
        [
            "--config",
            args.config,
            "--checkpoint",
            str(world_checkpoint),
            "--split",
            "train",
            "--resume",
            "--variant",
            args.variant,
            "--output-dir",
            str(trajectory_base),
            *optional("--max-videos", args.max_train_scenes),
        ],
    )
    export_validation = _torchrun(
        args.python_bin,
        args.master_addr,
        args.base_port + 1,
        args.num_gpus,
        "recipe.shared.evaluate.clevrer.export_qa_trajectories",
        [
            "--config",
            args.config,
            "--checkpoint",
            str(world_checkpoint),
            "--split",
            "validation",
            "--resume",
            "--variant",
            args.variant,
            "--output-dir",
            str(trajectory_base),
            *optional("--max-videos", val_export_scenes),
        ],
    )
    export_test = _torchrun(
        args.python_bin,
        args.master_addr,
        args.base_port + 2,
        args.num_gpus,
        "recipe.shared.evaluate.clevrer.export_qa_trajectories",
        [
            "--config",
            args.config,
            "--checkpoint",
            str(world_checkpoint),
            "--split",
            "test",
            "--resume",
            "--variant",
            args.variant,
            "--output-dir",
            str(trajectory_base),
            *optional("--max-videos", args.max_eval_scenes),
        ],
    )
    train_qa = _torchrun(
        args.python_bin,
        args.master_addr,
        args.base_port + 3,
        args.num_gpus,
        "recipe.shared.evaluate.clevrer.train_qa",
        [
            "--config",
            args.config,
            "--trajectory-root",
            str(trajectory_root),
            "--output-dir",
            str(qa_dir),
            *optional("--max-train-scenes", args.max_train_scenes),
            *optional("--max-val-scenes", args.max_val_scenes),
            *optional("--epochs", args.epochs),
        ],
    )
    eval_qa = _torchrun(
        args.python_bin,
        args.master_addr,
        args.base_port + 4,
        args.num_gpus,
        "recipe.shared.evaluate.clevrer.eval_qa",
        [
            "--config",
            args.config,
            "--checkpoint",
            str(qa_dir / "best.pt"),
            "--trajectory-root",
            str(trajectory_root),
            "--output-dir",
            str(qa_eval_dir),
            "--split",
            "validation",
            *optional("--max-scenes", args.max_eval_scenes),
        ],
    )
    test_qa = _torchrun(
        args.python_bin,
        args.master_addr,
        args.base_port + 5,
        args.num_gpus,
        "recipe.shared.evaluate.clevrer.eval_qa",
        [
            "--config",
            args.config,
            "--checkpoint",
            str(qa_dir / "best.pt"),
            "--trajectory-root",
            str(trajectory_root),
            "--output-dir",
            str(qa_test_dir),
            "--split",
            "test",
            "--submission-only",
            *optional("--max-scenes", args.max_eval_scenes),
        ],
    )

    try:
        _run_export_stage(
            "export_train",
            export_train,
            trajectory_root,
            "train",
            args.max_train_scenes,
            skip_existing_trajectories,
            primary_qa_root / f"qa_{args.variant}_export_train.log",
            None if mirror_qa_root is None else mirror_qa_root / f"qa_{args.variant}_export_train.log",
        )
        _run_export_stage(
            "export_validation",
            export_validation,
            trajectory_root,
            "validation",
            val_export_scenes,
            skip_existing_trajectories,
            primary_qa_root / f"qa_{args.variant}_export_validation.log",
            None if mirror_qa_root is None else mirror_qa_root / f"qa_{args.variant}_export_validation.log",
        )
        _run_export_stage(
            "export_test",
            export_test,
            trajectory_root,
            "test",
            args.max_eval_scenes,
            skip_existing_trajectories,
            primary_qa_root / f"qa_{args.variant}_export_test.log",
            None if mirror_qa_root is None else mirror_qa_root / f"qa_{args.variant}_export_test.log",
        )
        _run_stage(
            "train_qa",
            train_qa,
            primary_qa_root / f"qa_{args.variant}_train.log",
            None if mirror_qa_root is None else mirror_qa_root / f"qa_{args.variant}_train.log",
        )
        if not (qa_dir / "best.pt").is_file():
            raise FileNotFoundError("QA training did not produce best.pt")
        _run_stage(
            "eval_qa_validation",
            eval_qa,
            primary_qa_root / f"qa_{args.variant}_eval_validation.log",
            None if mirror_qa_root is None else mirror_qa_root / f"qa_{args.variant}_eval_validation.log",
        )
        if not (qa_eval_dir / "metrics.json").is_file():
            raise FileNotFoundError("QA evaluation did not produce metrics.json")
        if mirror_qa_root is not None:
            _mirror_lightweight_outputs(primary_qa_root, mirror_qa_root)
        _run_stage(
            "eval_qa_test",
            test_qa,
            primary_qa_root / f"qa_{args.variant}_eval_test.log",
            None if mirror_qa_root is None else mirror_qa_root / f"qa_{args.variant}_eval_test.log",
        )
        if not (qa_test_dir / "submission.json").is_file():
            raise FileNotFoundError("QA test did not produce submission.json")
        if mirror_qa_root is not None:
            _mirror_lightweight_outputs(primary_qa_root, mirror_qa_root)
        status = 0
        print(f"CLEVRER QA pipeline completed: {qa_eval_dir / 'metrics.json'}", flush=True)
    except BaseException:
        status = 1
        raise
    finally:
        primary_status_file.write_text(f"{status}\n", encoding="utf-8")
        if mirror_status_file is not None:
            mirror_status_file.write_text(f"{status}\n", encoding="utf-8")
            _mirror_lightweight_outputs(primary_qa_root, mirror_qa_root)


if __name__ == "__main__":
    main()
