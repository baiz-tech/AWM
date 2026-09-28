# Contributing

The normative specification for this repository is
[`docs/manual/requirement.md`](docs/manual/requirement.md). Read it before
changing anything; the summary below only highlights the points that are easy to
get wrong.

## Ground rules

- **Scope**: change only what the task needs. No drive-by refactors, no
  repository-wide formatting.
- **Source of paths**: data roots, checkpoints, devices and ports come from a
  config, CLI flag, environment variable, or `docs/manual/resource.md`. Never
  hard-code a machine-specific absolute path into shared source.
- **Errors**: never swallow an exception, never silently fall back, never fake a
  default so a run "looks successful". Fail early with an actionable message.
- **Reproducibility**: record the seed. Any new randomness must be made
  consistent across Python, NumPy, PyTorch, DataLoader workers and ranks.
- **Distributed**: check rank-local data, samplers, metric aggregation, sync
  points, and that only global rank 0 writes shared files.

## Layout rules

| Kind | Location |
|---|---|
| Experiment definition | `configs/<dataset>/<experiment>/config.yaml` |
| Launcher script | `scripts/<dataset>/<experiment>/<task>.sh` |
| Dataset-agnostic code | `src/core/` |
| Training / distributed runtime | `src/training/` |
| Dataset adapter, targets, cache | `src/data/<dataset>/` |
| Experiment implementation | `src/experiments/<experiment>/` |
| Analysis / study | `src/studies/<study>/` |
| Tests | `tests/` |

`<dataset>` is one of `physionpp`, `clevrer`, `ek100`. Experiment names use lower
case letters, digits and underscores only, and are used identically for the
config directory, the script directory and the `src/experiments/` package.

## Adding or changing a task

1. Put the implementation in `src/experiments/<experiment>/<task>.py` and make it
   startable with `python -m src.experiments.<experiment>.<task>`.
2. Read experiment parameters through `src/core/run_context.py`:

   ```python
   from src.core.run_context import apply_cli_defaults, task_context

   def main():
       ctx = task_context()
       parser = argparse.ArgumentParser()
       parser.add_argument("--output-dir", required=True)
       ...
       apply_cli_defaults(parser, ctx)
       args = parser.parse_args()
   ```

   `ctx` reads `CONFIG` / `TASK` / `LARGE_DATA_OUTPUT_DIR` /
   `LIGHT_DATA_OUTPUT_DIR` / `RESUME` / `CHECKPOINT` / `RANK` / `LOCAL_RANK` /
   `WORLD_SIZE`. Declare the argument values in the task config under
   `experiment.cli_defaults`, keyed by option string, using
   `{experiment_root}` / `{large_output_dir}` / `{cfg:<dotted.path>}` templates.
3. Take output locations from `ctx.large_dir` (checkpoints, caches, predictions)
   and `ctx.light_dir` (logs, metrics, summaries). Do **not** re-derive them from
   `workspace_root`, `data_root`, `output_mode` or `stage`.
4. Take resume state from `ctx.resume` and `ctx.checkpoint`; never scan
   directories for a checkpoint.
5. Declare upstream artifacts explicitly under `experiment.inputs` (or in
   `cli_defaults`) and read them with `ctx.input(...)` / `ctx.require_input(...)`;
   never discover them.
6. Only global rank 0 writes shared files (`ctx.is_main`).
7. Declare the task in the experiment config with its `module`, `distributed`,
   `output_mode`, `existing_output`, `background`, `log_name`, `pid_name`,
   `resume` and `args`.
8. Add the thin script:

   ```bash
   #!/usr/bin/env bash
   set -euo pipefail

   CONFIG=configs/<dataset>/<experiment>/config.yaml
   TASK=<task>

   source tools/launcher/launch_from_config.sh "$@"
   ```

## Before you hand work over

Run at least the checks that match the risk of your change:

```bash
python -m py_compile $(git ls-files '*.py')
python -m unittest discover -s tests -t . -p "test_*.py"
bash scripts/<dataset>/<experiment>/<task>.sh --dry-run
```

`--dry-run` must not create or modify outputs, pid files, locks or status files.

## Vendored code

`external/vjepa2/` is a vendored subset of upstream V-JEPA 2. Do not change its
algorithms. If upstream changes, re-extract the same file set and re-apply the
`external.vjepa2.` import prefix; see `external/README.md`.

## Data hygiene

- Keep train / validation / test strictly separated. Never tune on `test`.
- Check for video-level duplicates, overlapping clips and future-frame leakage
  before reporting a number.
- Report metric definition, aggregation, sample size and per-group results. Do
  not report only the favourable metric, and do not present correlation as
  causation.

## Version control

Do not commit generated outputs, checkpoints, caches, datasets or logs. Do not
rewrite or drop other people's uncommitted work.
