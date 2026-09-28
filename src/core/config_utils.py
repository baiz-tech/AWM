"""Helpers for loading task-scoped config blocks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def _path_from(value, default_base, bases):
    if value is None:
        return None
    if isinstance(value, dict):
        base_name = value.get("base")
        relative = value.get("path")
        if not base_name or relative is None:
            raise ValueError("path mappings must define both base and path")
        if base_name not in bases:
            raise ValueError(f"unknown path base: {base_name}")
        base = bases[base_name]
    else:
        relative = value
        base = default_base
    path = Path(str(relative))
    if path.is_absolute():
        return path
    return Path(base) / path


def _attach_run_paths(raw_config, config):
    launch = raw_config.get("launch") or {}
    outputs = launch.get("outputs") or {}
    run = launch.get("run") or {}
    config = deepcopy(config)
    workspace_root = Path(str(run.get("workspace_root", "outputs/runs")))
    data_root = Path(str(run.get("data_root", "/data/shyang/outputs")))
    root_bases = {
        "workspace_root": workspace_root,
        "data_root": data_root,
    }
    workspace_run_dir = _path_from(outputs.get("workspace_run_dir"), workspace_root, root_bases)
    data_run_dir = _path_from(outputs.get("data_run_dir"), data_root, root_bases)
    if workspace_run_dir is not None:
        config["folder"] = str(workspace_run_dir)
    elif run:
        config["folder"] = str(
            workspace_root
            / str(run.get("experiment", "vjepa2_naive"))
            / str(run.get("name", "run"))
        )
    if outputs:
        resolved_outputs = deepcopy(outputs)
        if workspace_run_dir is not None:
            resolved_outputs["workspace_run_dir"] = str(workspace_run_dir)
        if data_run_dir is not None:
            resolved_outputs["data_run_dir"] = str(data_run_dir)
        config.setdefault("outputs", {}).update(resolved_outputs)
    return config


def _legacy_task_experiment(raw_config, task):
    experiment = raw_config.get("experiment")
    if experiment is None:
        return deepcopy(raw_config)
    if task == "train":
        return deepcopy(experiment) if isinstance(experiment, dict) else {}
    if isinstance(experiment, dict):
        if task == "qa_eval" and "qa" in experiment:
            return {"qa": deepcopy(experiment.get("qa") or {})}
        if task == "full_future_ocp" and "evaluation" in experiment:
            return {"evaluation": deepcopy(experiment.get("evaluation") or {})}
    return deepcopy(experiment) if isinstance(experiment, dict) else {}


def task_launch(raw_config, task):
    launch = raw_config.get("launch") or {}
    tasks = raw_config.get("tasks") or {}
    task_entry = tasks.get(task) or {}
    launch_spec = deepcopy(task_entry.get("launch") or {})
    if launch_spec:
        return launch_spec
    if task == "train":
        legacy = (launch.get("tasks") or {}).get(task) or {}
    else:
        legacy = (launch.get("evaluations") or {}).get(task) or {}
    return deepcopy(legacy)


def task_experiment(raw_config, task):
    tasks = raw_config.get("tasks") or {}
    task_entry = tasks.get(task) or {}
    experiment = task_entry.get("experiment")
    if experiment is not None:
        return _attach_run_paths(raw_config, experiment)
    return _attach_run_paths(raw_config, _legacy_task_experiment(raw_config, task))


def training_experiment(raw_config):
    return task_experiment(raw_config, "train")


def merged_task_experiment(raw_config, task, base_task="train"):
    base = task_experiment(raw_config, base_task)
    if not isinstance(base, dict):
        base = {}
    task_specific = task_experiment(raw_config, task)
    if not isinstance(task_specific, dict):
        task_specific = {}
    merged = deepcopy(base)
    for key, value in task_specific.items():
        merged[key] = deepcopy(value)
    return merged
