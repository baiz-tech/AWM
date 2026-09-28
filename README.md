# Abductive World Modeling (AWM)

Code for **Abductive World Modeling via Causal Representation Learning**.

The framework predicts a future latent with a frozen predictive video backbone and
then **abduces backward** from that predicted future together with the current
observation to infer a structured world state. The structure is realised by the
**Hierarchical Abductive State Pyramid (HASP)**, which organises the inferred
state into three levels:

```text
Entity    what exists          object-centred slots
Dynamic   how it changes       slot x time states
Relation  how they interact    entity-pair x time states
```

Downstream readouts consume these states for physical prediction (Physion++),
casual/event reasoning (CLEVRER), and action understanding (EK100).

## Repository layout

```text
configs/<dataset>/<experiment>/config.yaml   experiment definition (launch + tasks)
scripts/<dataset>/<experiment>/<task>.sh     thin launchers (CONFIG + TASK only)
configs/analyses/<analysis>/config.yaml      cross-run analyses (read frozen artifacts only)
scripts/analyses/<analysis>/<task>.sh
src/core/                                    dataset-agnostic models, losses, interfaces
src/training/                                distributed runtime, checkpoint, logging
src/data/<dataset>/                          dataset adapters, targets, caches, QA protocol
src/experiments/<experiment>/                experiment implementations
src/studies/<study>/                         analyses, diagnostics, ablations
tests/                                       unit and launcher-contract tests
docs/experiment.md                           how to run every experiment (start here)
docs/manual/                                 layout/config specification + machine resources
external/vjepa2/                             vendored upstream V-JEPA 2 subset
tools/launcher/                              shared launcher
legacy/                                      superseded experiments and scripts, kept verbatim
```

Documentation:

| file | content |
|---|---|
| [`docs/experiment.md`](docs/experiment.md) | how to run each experiment, the dependency chains, and which task produces which paper table / figure |
| [`docs/manual/requirement.md`](docs/manual/requirement.md) | governing specification: layout, naming, config schema, launcher/task contract |
| [`docs/manual/example_config.yaml`](docs/manual/example_config.yaml) | canonical config example with every supported field |
| [`docs/manual/resource.md`](docs/manual/resource.md) | measured resources of this machine: data, checkpoints, GPU state, ports |
| [`tools/launcher/README.md`](tools/launcher/README.md) | launcher contract and template variables |

## Environment

```bash
/data/shared/envs/vjepa2-312/bin/python    # Python 3.12 + torch 2.6.0+cu124
```

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Code under `external/vjepa2/` is imported as `external.vjepa2.*`; no `PYTHONPATH`
tweaks are needed as long as commands are run from the repository root.

## Running an experiment

Every task is a thin script that only selects the config and the task name and then
delegates to `tools/launcher/launch_from_config.sh`:

```bash
# Inspect the fully resolved configuration without side effects.
bash scripts/physionpp/awm_physionpp_fullpatch_probe_seed239/train_predictor.sh --dry-run

# Start it in the background (the default).
bash scripts/physionpp/awm_physionpp_fullpatch_probe_seed239/train_predictor.sh

# Choose an output location.
bash scripts/clevrer/awm_clevrer_fullpatch_probe_seed239/train_probe.sh --output-mode workspace
```

The launcher resolves the config, exports the `CONFIG` / `TASK` /
`LARGE_DATA_OUTPUT_DIR` / `LIGHT_DATA_OUTPUT_DIR` / `RESUME` / `CHECKPOINT` contract
plus the `torchrun` rank variables, and records `logs/`, `configs/` (including a
task-resolved config), `manifest.json`, `command.txt` and `environment.json`.

The task module itself turns `tasks.<task>.experiment.cli_defaults` into its
argument defaults through `src/core/run_context.py`, so the launcher passes no
arguments and a module can also be started by hand with the same environment.
An explicit CLI argument always overrides the config.

**Step-by-step instructions for every experiment — training, baselines, controls
and the offline analysis / intervention experiments — are in
[`docs/experiment.md`](docs/experiment.md).** It also lists the dependency chain of
each experiment, so read it before starting anything. The launcher contract itself
is documented in [`tools/launcher/README.md`](tools/launcher/README.md).

## Experiments

| dataset | experiment | what it is |
|---|---|---|
| physionpp | `awm_physionpp_fullpatch_probe_seed239` | AWM / HASP main Physion++ experiment |
| clevrer | `awm_clevrer_fullpatch_probe_seed239` | AWM / HASP main CLEVRER experiment |
| clevrer | `awm_clevrer_current_only_seed239` | current-latent-only predictability control |
| ek100 | `awm_ek100_multiscale_adapter_seed239` | AWM / HASP main EK100 experiment |
| ek100 | `awm_ek100_fair_attribution_seed239` | parameter-matched attribution matrix (E0–E4) |
| physionpp | `vjepa2_physionpp_dynamics_decoder_seed239` | V-JEPA 2 matched baseline |
| clevrer | `vjepa2_clevrer_naive_world_model_seed239` | V-JEPA 2 matched baseline (paper Table 1) |
| clevrer | `vjepa2_clevrer_dynamics_decoder_seed239` | V-JEPA 2 decoder baseline |
| clevrer | `vjepa2_clevrer_dynamics_decoder_all_seed239` | V-JEPA 2 all-future decoder baseline |
| ek100 | `vjepa2_ek100_dynamics_decoder_seed239` | V-JEPA 2 decoder baseline |
| physionpp | `videomae2_physionpp_oracle_ocp_seed239` | VideoMAE v2 baseline (ground-truth future) |
| ek100 | `videomae2_ek100_finetune_seed239` | VideoMAE v2 baseline (fine-tuned encoder) |
| physionpp | `orca_physionpp_oracle_ocp_seed239` | Orca baseline (ground-truth future) |
| ek100 | `orca_ek100_frozen_readout_seed239` | Orca baseline (frozen encoder) |

Offline analyses and ablations live under `src/studies/`:
`hasp_offline_analysis`, `clevrer_intervention`, `clevrer_relation_controls`,
`paper_figures`, `clevrer_interaction_conditioned_future`,
`object_slot_generalization/{v1,v2,v3}`, `fair_attribution`.

Four of them are wired through their own configs under `configs/analyses/`:
`hasp_offline_analysis`, `clevrer_intervention` (paper table `query_intervention`),
`clevrer_relation_controls` (figure `interaction_prediction`) and `paper_figures`
(the two data-driven figures; requires `matplotlib`).

## Tests

`pytest` is not installed in the `vjepa2-312` environment, so the unittest-based
tests run with the standard library runner:

```bash
cd makeup
python -m unittest discover -s tests -t . -p "test_*.py"
```

A handful of test modules are written in pytest style (bare `test_*` functions).
They are executed by `tools/run_pytest_style_tests.py` until pytest is available:

```bash
python tools/run_pytest_style_tests.py
```

## Status of this repository

This tree was produced by reorganising an internal working repository. 18 configs
and 103 launcher tasks are wired end to end: every task resolves its full command
line from its config, and `tools/check_task_defaults.py` plus a `--dry-run` sweep
over every `scripts/**/*.sh` pass on CPU.

What has **not** been done: no GPU run has been executed in this environment, the
EK100 / Orca data and weights are absent here, and the paper's reported numbers
have not been re-verified. See [`docs/experiment.md`](docs/experiment.md) §12 for
the exact list of what cannot run on this machine.

## External dependencies that are not vendored

- `external/vjepa2/` — a vendored subset of V-JEPA 2 (licence at
  `external/vjepa2/LICENSE`).
- OpenGVLab/VideoMAEv2 — needed only by the two `videomae2_*` experiments. Clone
  it and point `model.official_repo` at the clone:
  `git clone https://github.com/OpenGVLab/VideoMAEv2 external/VideoMAEv2`.
  `model.checkpoint` must point at the official `.pth` weights. The loader fails
  loudly when either path is missing.

## License

This repository is released under the Apache License 2.0 — see [`LICENSE`](LICENSE).
The vendored upstream V-JEPA 2 code under `external/vjepa2/` retains its upstream
license, included at `external/vjepa2/LICENSE`.
