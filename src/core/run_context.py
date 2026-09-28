"""Runtime contract shared by every task module.

`docs/manual/requirement.md` section 5.1 fixes what a task module may rely on at
run time. It receives everything through the environment, injected by
``tools/launcher/launch_from_config.sh``:

===============  =========================================================
``CONFIG``       the experiment config actually in use
``TASK``         the task name, e.g. ``train`` / ``eval``
``LARGE_DATA_OUTPUT_DIR``  checkpoints, caches, predictions
``LIGHT_DATA_OUTPUT_DIR``  logs, metrics, summaries
``RESUME``       ``1`` when the launcher resolved a resume run
``CHECKPOINT``   the checkpoint the launcher resolved; never re-discovered here
``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``  ``torchrun`` semantics
===============  =========================================================

What a task module must **not** do, and what this module therefore never does:

* derive its own output directory from ``workspace_root`` / ``data_root`` /
  ``output_mode`` / ``stage`` -- output paths come from ``LARGE_DATA_OUTPUT_DIR``
  and ``LIGHT_DATA_OUTPUT_DIR`` only;
* scan directories or compare mtimes to find a checkpoint;
* guess the current task name.

Upstream artifacts that one task consumes from another (a cache, a probe
checkpoint, a target file) are *inputs*: they are declared explicitly in the
config under ``tasks.<task>.experiment.inputs`` and the module reads them from
there. They are never discovered.

Outside the launcher (a manual ``python -m ...`` invocation, or a unit test) the
environment is absent. In that case ``RunContext.from_env()`` returns a context
whose fields are ``None``, and each ``argparse`` default falls back to the
explicit CLI argument, so debugging a single module still works.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.core.config_utils import task_experiment

VARIABLE_RE = re.compile(r"\{([^{}]+)\}")

ENV_CONFIG = "CONFIG"
ENV_TASK = "TASK"
ENV_LARGE = "LARGE_DATA_OUTPUT_DIR"
ENV_LIGHT = "LIGHT_DATA_OUTPUT_DIR"
ENV_RESUME = "RESUME"
ENV_CHECKPOINT = "CHECKPOINT"


def _optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - launcher always sets ints
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass(frozen=True)
class RunContext:
    """Resolved runtime information for one task."""

    task: str | None = None
    config_path: Path | None = None
    large_dir: Path | None = None
    light_dir: Path | None = None
    resume: bool = False
    checkpoint: Path | None = None
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    experiment: Mapping[str, Any] = field(default_factory=dict)
    raw_config: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RunContext":
        env = os.environ if env is None else env
        config_path = _optional_path(env.get(ENV_CONFIG))
        task = env.get(ENV_TASK) or None

        experiment: Mapping[str, Any] = {}
        raw_config: Mapping[str, Any] = {}
        if config_path is not None and task is not None:
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"CONFIG points at a missing file: {config_path}"
                )
            raw_config = _load_yaml(config_path)
            tasks = raw_config.get("tasks") or {}
            if task not in tasks:
                raise KeyError(
                    f"task {task!r} is not declared in {config_path} "
                    f"(declared: {sorted(tasks)})"
                )
            experiment = task_experiment(raw_config, task) or {}

        return cls(
            task=task,
            config_path=config_path,
            large_dir=_optional_path(env.get(ENV_LARGE)),
            light_dir=_optional_path(env.get(ENV_LIGHT)),
            resume=(env.get(ENV_RESUME) == "1"),
            checkpoint=_optional_path(env.get(ENV_CHECKPOINT)),
            rank=_int_env("RANK", 0),
            local_rank=_int_env("LOCAL_RANK", 0),
            world_size=_int_env("WORLD_SIZE", 1),
            experiment=experiment,
            raw_config=raw_config,
        )

    def block(self, name: str) -> Mapping[str, Any]:
        """Return ``tasks.<name>.experiment`` from the raw config.

        Several modules need a *shared protocol* block rather than their own,
        e.g. the CLEVRER cache reads ``tasks.train.experiment`` for the world
        model and data protocol. The block is declared in the config; nothing is
        inferred from the environment.
        """
        entry = (self.raw_config.get("tasks") or {}).get(name) or {}
        experiment = entry.get("experiment")
        if experiment is None:
            raise KeyError(
                f"tasks.{name}.experiment is required but not declared in "
                f"{self.config_path}"
            )
        return experiment


    @property
    def launched(self) -> bool:
        """True when running under the launcher."""
        return self.large_dir is not None and self.light_dir is not None

    @property
    def experiment_root(self) -> Path | None:
        """The experiment-level directory: the parent of this task's output dir."""
        return None if self.large_dir is None else self.large_dir.parent

    @property
    def resolved_config(self) -> Path | None:
        """Where the launcher wrote this task's resolved config."""
        if self.light_dir is None or self.task is None:
            return None
        return self.light_dir / "configs" / f"{self.task}.resolved.yaml"


    @property
    def is_main(self) -> bool:
        """True on the global rank that owns shared files."""
        return self.rank == 0

    # --------------------------------------------------------------- lookups
    def raw(self, dotted: str, default: Any = None) -> Any:
        """Dotted lookup inside ``tasks.<task>.experiment``."""
        node: Any = self.experiment
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def get(self, dotted: str, default: Any = None) -> Any:
        """Like :meth:`raw`, but renders path templates first when it is a str."""
        value = self.raw(dotted, default)
        if isinstance(value, str):
            return self.render(value)
        return value

    def require(self, dotted: str) -> Any:
        """Like :meth:`get`, but fails with an actionable message when absent."""
        value = self.get(dotted, None)
        if value is None:
            raise KeyError(
                f"tasks.{self.task}.experiment.{dotted} is required but not set "
                f"in {self.config_path}"
            )
        return value

    def input(self, name: str, default: Any = None) -> Any:
        """Read a declared upstream artifact from ``experiment.inputs``."""
        return self.get(f"inputs.{name}", default)

    def require_input(self, name: str) -> Any:
        value = self.input(name, None)
        if value is None:
            raise KeyError(
                f"tasks.{self.task}.experiment.inputs.{name} is required but not "
                f"set in {self.config_path}"
            )
        return value

    def output(self, name: str = "", default: Any = None) -> Any:
        """Read a declared output location from ``experiment.outputs``.

        Falls back to ``LARGE_DATA_OUTPUT_DIR`` (optionally joined with *name*),
        which is the only output root a task module is allowed to assume.
        """
        declared = self.get(f"outputs.{name}", None) if name else None
        if declared is not None:
            return declared
        base = default if default is not None else self.large_dir
        if base is None:
            return None
        return Path(base) / name if name else Path(base)

    # -------------------------------------------------------------- rendering
    def render(self, value: str) -> str:
        """Expand the path templates allowed inside a task experiment block.

        The launcher uses the same vocabulary in ``tasks.<task>.launch.args``;
        keeping it identical means a value can move between the two without
        being rewritten.
        """

        def replace(match: re.Match) -> str:
            body = match.group(1)
            if body.startswith("cfg:"):
                node: Any = self.raw_config
                for part in body[4:].split("."):
                    if isinstance(node, Mapping) and part in node:
                        node = node[part]
                    else:
                        raise KeyError(
                            f"tasks.{self.task}.experiment.cli_defaults uses "
                            f"{{{body}}}, which does not resolve in "
                            f"{self.config_path}"
                        )
                return "" if node is None else str(node)
            known = {
                "config": self.config_path,
                "task": self.task,
                "large_output_dir": self.large_dir,
                "light_output_dir": self.light_dir,
                "experiment_root": self.experiment_root,
                "resolved_config": self.resolved_config,
                "checkpoint": self.checkpoint,
                "seed": self.raw("seed"),
            }
            if body not in known:
                return match.group(0)
            resolved = known[body]
            if resolved is None:
                return match.group(0)
            return str(resolved)

        return VARIABLE_RE.sub(replace, value)

    # ------------------------------------------------------------------ guard
    def require_launched(self, what: str) -> None:
        """Fail early when a value can only come from the launcher."""
        if not self.launched:
            raise RuntimeError(
                f"{what} is not available: this module must be started through "
                f"tools/launcher/launch_from_config.sh, or the corresponding "
                f"CLI argument must be given explicitly "
                f"(missing {ENV_LARGE}/{ENV_LIGHT})"
            )


def task_context() -> RunContext:
    """Convenience wrapper used by task modules at import/parse time."""
    return RunContext.from_env()


def apply_cli_defaults(parser, ctx: RunContext, options: set[str] | None = None):
    """Fill an ``argparse`` parser's defaults from the task's config block.

    Every task module declares its runtime defaults under
    ``tasks.<task>.experiment.cli_defaults``, keyed by the exact option string::

        experiment:
          cli_defaults:
            --output-dir: "{large_output_dir}"
            --checkpoint: "{experiment_root}/train_probe/best.pt"
            --epochs: "30"

    Why here and not in the module: the value of an argument is a property of
    the *experiment*, not of the code, and requirement.md section 5 puts paths,
    devices and hyper-parameters in the config. Keeping them in one place also
    makes a run reproducible from the config alone.

    An explicit CLI argument still wins, so a module stays debuggable with a
    manual ``python -m ... --output-dir /tmp/x``.

    Unknown option names fail loudly rather than being ignored.

    ``options`` restricts the application to a subset and disables the unknown
    check for the rest. It exists for two-phase parsers, where a preliminary
    parser must receive only ``--config`` before the full parser is built.
    """
    import argparse

    declared = ctx.raw("cli_defaults") or {}
    if not declared:
        return parser
    if options is not None:
        declared = {key: value for key, value in declared.items() if key in options}
        if not declared:
            return parser

    known = {
        option: action
        for action in parser._actions
        for option in action.option_strings
    }
    if options is None:
        unknown = sorted(set(declared) - set(known))
        if unknown:
            raise KeyError(
                f"tasks.{ctx.task}.experiment.cli_defaults declares options that "
                f"{parser.prog} does not accept: {unknown}"
            )
    declared = {key: value for key, value in declared.items() if key in known}

    resolved: dict[str, Any] = {}
    for option, raw_value in declared.items():
        action = known[option]
        value = ctx.render(raw_value) if isinstance(raw_value, str) else raw_value
        # A placeholder that survived rendering means the launcher environment is
        # absent (manual invocation). Keep the CLI requirement in place so the
        # failure names the missing option instead of passing a literal "{...}".
        if isinstance(value, str) and VARIABLE_RE.search(value):
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes", "on"}
            value = bool(value)
        elif isinstance(action, argparse._StoreFalseAction):
            if isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes", "on"}
            value = not bool(value)
        elif action.type is not None and isinstance(value, str):
            value = action.type(value)
        elif action.nargs in ("+", "*") and isinstance(value, str):
            value = value.split()
        if action.required:
            # The value now comes from the config/environment, so argparse must
            # no longer demand it on the command line.
            action.required = False
        resolved[action.dest] = value

    parser.set_defaults(**resolved)
    return parser


def require_cli(parser, **named):
    """Fail with an actionable message when a value is only available via CLI.

    Used after ``parse_args`` for arguments that have no config default, so the
    failure names the option instead of raising ``AttributeError`` deep inside a
    training loop.
    """
    missing = [option for option, value in named.items() if value is None]
    if missing:
        parser.error(
            "the following arguments are required when the task is not started "
            "through tools/launcher/launch_from_config.sh: "
            + ", ".join(sorted(missing))
        )



def resolve_resume(ctx: RunContext, cli_resume: str | None) -> Path | None:
    """Resolve ``--resume`` against the launcher-provided checkpoint.

    ``requirement.md`` section 5.1: whether this run is a resume is decided by
    the launcher (``RESUME``), and ``CHECKPOINT`` is the path it resolved. A task
    module must not scan for one.
    """
    if cli_resume:
        path = Path(cli_resume)
        if not path.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {path}")
        return path
    if not ctx.resume:
        return None
    if ctx.checkpoint is None:
        raise RuntimeError(
            "RESUME=1 but the launcher did not export CHECKPOINT; start the task "
            "through tools/launcher/launch_from_config.sh --resume"
        )
    if not ctx.checkpoint.is_file():
        raise FileNotFoundError(
            f"launcher-provided CHECKPOINT does not exist: {ctx.checkpoint}"
        )
    return ctx.checkpoint
