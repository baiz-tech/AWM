#!/usr/bin/env python3
"""Render one task of a launch config for ``launch_from_config.sh``.

Two responsibilities:

1. Build the *task-resolved config* that the task module actually consumes. It is
   either ``tasks.<task>.experiment`` or an optional ``launch.config_template``
   file, with the run paths injected and ``launch.config_overrides`` applied.
2. Interpolate ``tasks.<task>.launch.args`` / ``launch.env`` so that concrete
   paths are passed on the command line instead of being guessed by the module.

Modes
-----
``--no-write``  print the interpolated argv only (used by ``--dry-run`` and to
                assemble the command before any output directory is touched).
``--write``     additionally persist the resolved config and the rendered env
                file, and print the argv.

Template variables (available in ``args``, ``env`` values and override values)::

    {config} {resolved_config} {task} {project} {dataset} {experiment} {stage}
    {experiment_root} {data_experiment_root} {workspace_experiment_root}
    {large_output_dir} {light_output_dir} {checkpoint} {seed}
    {nnodes} {nproc} {master_addr} {master_port}
    {cfg:<dotted.path>}      value looked up in the raw config
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from pathlib import Path

import yaml

VARIABLE_RE = re.compile(r"\{([^{}]+)\}")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_context(config: dict, config_path: str, task: str, env: dict) -> dict:
    run = (config.get("launch") or {}).get("run") or {}
    task_entry = (config.get("tasks") or {}).get(task) or {}
    experiment = task_entry.get("experiment")
    seed = ""
    if isinstance(experiment, dict) and experiment.get("seed") is not None:
        seed = str(experiment["seed"])

    return {
        "config": config_path,
        "resolved_config": env.get("LAUNCH_RESOLVED_CONFIG", ""),
        "task": task,
        "project": str(run.get("project", "")),
        "dataset": str(run.get("dataset", "")),
        "experiment": str(run.get("experiment", "")),
        "stage": str(run.get("stage", 0)),
        "experiment_root": env.get("LAUNCH_EXPERIMENT_ROOT", ""),
        "data_experiment_root": env.get("LAUNCH_DATA_EXPERIMENT_ROOT", ""),
        "workspace_experiment_root": env.get("LAUNCH_WORKSPACE_EXPERIMENT_ROOT", ""),
        "large_output_dir": env.get("LARGE_DATA_OUTPUT_DIR", ""),
        "light_output_dir": env.get("LIGHT_DATA_OUTPUT_DIR", ""),
        "checkpoint": env.get("CHECKPOINT", ""),
        "seed": seed,
        "nnodes": env.get("LAUNCH_NNODES", ""),
        "nproc": env.get("LAUNCH_NPROC_PER_NODE", ""),
        "master_addr": env.get("LAUNCH_MASTER_ADDR", ""),
        "master_port": env.get("LAUNCH_MASTER_PORT", ""),
    }


def config_lookup(config: dict, dotted: str):
    node = config
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def make_renderer(config: dict, context: dict):
    def render(value):
        if not isinstance(value, str):
            return value

        def replace(match: re.Match) -> str:
            body = match.group(1)
            if body.startswith("cfg:"):
                found = config_lookup(config, body[4:])
                if found is None:
                    raise SystemExit(
                        f"template {{{body}}} does not resolve in the config"
                    )
                return str(found)
            if body in context:
                resolved = context[body]
                if resolved == "" and body in {"experiment_root", "large_output_dir",
                                               "light_output_dir", "resolved_config"}:
                    raise SystemExit(
                        f"template {{{body}}} is empty; the launcher must set it"
                    )
                return resolved
            raise SystemExit(f"unknown template variable {{{body}}}")

        return VARIABLE_RE.sub(replace, value)

    return render


def apply_overrides(resolved: dict, overrides: dict, render) -> None:
    for dotted, raw_value in overrides.items():
        node = resolved
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = render(raw_value)


def build_resolved(config: dict, task: str, context: dict, render) -> dict:
    task_entry = (config.get("tasks") or {}).get(task) or {}
    launch = task_entry.get("launch") or {}
    experiment = task_entry.get("experiment")

    template = launch.get("config_template")
    if template:
        template_path = Path(render(template))
        if not template_path.is_absolute():
            template_path = Path.cwd() / template_path
        if not template_path.is_file():
            raise SystemExit(f"tasks.{task}.launch.config_template not found: {template_path}")
        resolved = load_config(str(template_path))
    elif isinstance(experiment, dict):
        resolved = copy.deepcopy(experiment)
    else:
        resolved = {}

    resolved["folder"] = context["large_output_dir"]
    resolved["launch"] = {
        **(resolved.get("launch") or {}),
        "project": context["project"],
        "dataset": context["dataset"],
        "experiment": context["experiment"],
        "stage": context["stage"],
        "task": task,
        "task_output_dir": context["large_output_dir"],
        "experiment_root": context["experiment_root"],
        "data_experiment_root": context["data_experiment_root"],
        "workspace_experiment_root": context["workspace_experiment_root"],
    }
    apply_overrides(resolved, launch.get("config_overrides") or {}, render)

    # Compatibility mirror. Several modules were written against the older
    # config layout and read their parameters as
    #   config["tasks"]["train"]["experiment"]      (direct dict access)
    #   training_experiment(config) / task_experiment(config, name)
    # (see src/core/config_utils.py). The resolved config therefore exposes the
    # same block under those keys as well, so a single file works for every
    # reader without touching module code.
    alias = str(launch.get("config_alias_task", "train"))
    body = {key: value for key, value in resolved.items() if key != "tasks"}
    resolved["tasks"] = {
        alias: {"experiment": copy.deepcopy(body)},
        task: {"experiment": copy.deepcopy(body)},
    }
    return resolved


def write_outputs(resolved: dict, rendered_env: dict, context: dict, task: str) -> None:
    config_path = Path(context["resolved_config"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    os.replace(tmp, config_path)

    if rendered_env:
        env_path = config_path.parent / f"{task}.env"
        tmp = env_path.with_suffix(env_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            for key, value in rendered_env.items():
                handle.write(f"export {key}={shlex_quote(str(value))}\n")
        os.replace(tmp, env_path)


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    context = build_context(config, str(Path(args.config).resolve()), args.task, os.environ)
    render = make_renderer(config, context)

    task_launch = ((config.get("tasks") or {}).get(args.task) or {}).get("launch") or {}
    resolved = build_resolved(config, args.task, context, render)
    rendered_env = {
        key: render(value) for key, value in (task_launch.get("env") or {}).items()
    }

    if args.write:
        write_outputs(resolved, rendered_env, context, args.task)

    for arg in task_launch.get("args") or []:
        rendered = render(arg)
        if rendered is None or rendered == "":
            raise SystemExit(
                f"tasks.{args.task}.launch.args contains an empty value after rendering"
            )
        sys.stdout.buffer.write(str(rendered).encode("utf-8") + b"\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
