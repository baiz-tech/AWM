# Paper figures (AWM / HASP)

Reproduction of the two **data-driven** paper figures. The renderers here read
metric JSONs that upstream experiments already wrote; they never recompute
metrics, never impute missing values and never fall back to defaults.

## Figure inventory

| Paper label | Output PDF | Reproduced here? | Source of numbers |
| --- | --- | --- | --- |
| `fig:main` | `Ilustration_awm.pdf` | **No** — hand-drawn architecture diagram | none |
| `fig:entity-grounding` | `entity_grounding_ego_1/2/3.pdf`, `entity_grounding_temporal.pdf` | **No** — qualitative video-frame visualisations | none (not reducible to stored numbers) |
| `fig:entity-intervention` | `entity_intervention.pdf` | **Yes** — `render_entity_intervention.py` | OCP intervention JSON, protocol `physionpp3_ocp_intervention_v1` |
| `fig:relation-component` | `interaction_prediction.pdf` | **Yes** — `render_interaction_prediction.py` | CLEVRER relation-controls JSON, protocol `clevrer_relation_controls_v1` |

The architecture diagram and the four `entity_grounding_*` overlays are produced
by hand / from qualitative frame overlays and are deliberately **not** generated
by this package.

## Scripts

| Script | Paper figure | Output filename |
| --- | --- | --- |
| `render_entity_intervention.py` | `fig:entity-intervention` | `entity_intervention.pdf` |
| `render_interaction_prediction.py` | `fig:relation-component` | `interaction_prediction.pdf` |
| `render_all.py` | both of the above | both files, into one directory |

`render_all.py` is a thin wrapper; it requires both metric files to exist and
fails on the first missing or malformed input rather than skipping a figure.

## Prerequisites

**The upstream runs must have completed first.** Neither renderer can compute
anything on its own; each one only plots numbers that already exist in a JSON
file. Both renderers need `matplotlib` (see below).

The recommended entry point is the launcher task, which reads the two input paths
from `configs/analyses/paper_figures/config.yaml`:

```bash
cd makeup
bash scripts/analyses/paper_figures/render_all.sh --dry-run   # 先看解析后的输入与输出路径
bash scripts/analyses/paper_figures/render_all.sh
```

Point `launch.extra_config.paths.{physion_ocp_interventions,
clevrer_relation_controls_metrics}` at your own runs if they are not in the
default locations. `docs/experiment.md` §9.4 describes the same task in context.

### 1. `fig:entity-intervention` — OCP intervention JSON

Produced by the Physion++ V-JEPA 2 experiment's `evaluate_ocp_interventions` task
(protocol `physionpp3_ocp_intervention_v1`):

```bash
cd makeup
bash scripts/physionpp/vjepa2_physionpp_dynamics_decoder_seed239/evaluate_ocp_interventions.sh
# 前置:prepare_targets* → cache_latents* → train_probe → train_ocp_visual_only → train_ocp_readout
```

Expected JSON layout (only `metrics` is read):

```jsonc
{
  "protocol": "physionpp3_ocp_intervention_v1",
  "metrics": {
    "visual_only":     {"accuracy": 0.6489, "balanced_accuracy": 0.6401, "auroc": 0.7102, "...": "..."},
    "visual_probe":    {"accuracy": 0.7175, "balanced_accuracy": 0.7089, "auroc": 0.7962},
    "target_mask":     {"accuracy": 0.6325, "balanced_accuracy": 0.6211, "auroc": 0.7391},
    "irrelevant_mask": {"accuracy": 0.7088, "balanced_accuracy": 0.6992, "auroc": 0.7834},
    "random_slot":     {"accuracy": 0.6912, "balanced_accuracy": 0.6830, "auroc": 0.7698}
  },
  "paired_delta_vs_visual_probe": {"...": "..."}
}
```

All five conditions are optional; only the ones present in the JSON are drawn.
`accuracy`, `balanced_accuracy` and `auroc` are required for every condition
that is drawn, and each must be a real number (`null` is rejected).  The
paper's "Full" condition is the unmasked `visual_probe` baseline.

### 2. `fig:relation-component` — CLEVRER relation controls

Produced by the `clevrer_relation_controls` analysis (protocol
`clevrer_relation_controls_v1`):

```bash
cd makeup
bash scripts/analyses/clevrer_relation_controls/extract_shards.sh   # 8 卡导出 pair 级特征
bash scripts/analyses/clevrer_relation_controls/merge_shards.sh     # 合并并写出 metrics.json
```

`merge_shards` reads the shard directory written by `extract_shards`, so the two
tasks must be run in that order. To invoke the module directly instead:

```bash
cd makeup
PYTHONPATH="$PWD" python -m src.studies.clevrer_relation_controls.run \
  --train-latents /path/to/latents/train \
  --validation-latents /path/to/latents/validation \
  --train-targets /path/to/targets_train.pt \
  --validation-targets /path/to/targets_validation.pt \
  --checkpoint /path/to/structured_probe/best.pt \
  --output /path/to/clevrer_relation_controls/metrics.json
```

Expected JSON layout (only `metrics` is read):

```jsonc
{
  "protocol": "clevrer_relation_controls_v1",
  "train_pairs": 40,
  "validation_pairs": 16,
  "checkpoint": "/path/to/structured_probe/best.pt",
  "metrics": {
    "Entity":            {"contact": {"auroc": 0.8102}, "ttc": {"mae": 0.2013}, "first_contact": {"...": "..."}, "feature_dim": 32},
    "Dynamic":           {"contact": {"auroc": 0.8331}, "ttc": {"mae": 0.1811}},
    "Relation-only":     {"contact": {"auroc": 0.9673}, "ttc": {"mae": 0.1054}},
    "Entity+Dynamic":    {"contact": {"auroc": 0.8602}, "ttc": {"mae": 0.1704}},
    "Entity+Relation":   {"contact": {"auroc": 0.9651}, "ttc": {"mae": 0.1062}},
    "Dynamic+Relation":  {"contact": {"auroc": 0.9662}, "ttc": {"mae": 0.1058}},
    "Full":              {"contact": {"auroc": 0.9680}, "ttc": {"mae": 0.1049}}
  }
}
```

The combination names and their order mirror the `VARIANTS` mapping in
`run.py`.  Every combination present in the JSON is drawn (a subset is fine);
`metrics[<combination>]["contact"]["auroc"]` and
`metrics[<combination>]["ttc"]["mae"]` are required and must be real numbers.
Contact AUROC (higher is better, left axis, 0–1) and TTC MAE (lower is better,
right axis) are drawn as grouped bars per combination.

## Usage

```bash
cd makeup
export PYTHONPATH="$PWD"

# both figures at once
python -m src.studies.paper_figures.render_all \
  --entity-intervention-metrics /path/to/ocp_interventions.json \
  --interaction-metrics /path/to/clevrer_relation_controls/metrics.json \
  --output-dir /path/to/figures

# or one at a time
python -m src.studies.paper_figures.render_entity_intervention \
  --metrics /path/to/ocp_interventions.json \
  --output /path/to/figures/entity_intervention.pdf [--title "..."]
python -m src.studies.paper_figures.render_interaction_prediction \
  --metrics /path/to/clevrer_relation_controls/metrics.json \
  --output /path/to/figures/interaction_prediction.pdf [--title "..."]
```

The output directory is created if needed.  No path is hardcoded: everything
comes from the CLI (or from the launcher env when started through the task).

## Failure behaviour

Both renderers raise instead of degrading:

- `FileNotFoundError` — the metrics file does not exist.
- `KeyError` — the top-level `metrics` object, a condition/combination, or a
  required score key is absent; also when the JSON contains none of the known
  conditions/combinations.
- `TypeError` — a container that should be an object is not.
- `ValueError` — a required score is present but not a real number
  (`null`, string, boolean).
- `ImportError` — `matplotlib` cannot be imported (see below).

Input validation runs before the plotting backend is imported, so schema
errors are reported even in environments without `matplotlib`.

## matplotlib

`matplotlib` is listed in `requirements.txt` but is **not** installed in the
`vjepa2-312` environment used by this checkout (only `matplotlib-inline` is).
The renderers therefore import it lazily and raise

```
ImportError: matplotlib is required to render the paper figures but is not importable
from the active interpreter. Install it with `python -m pip install matplotlib`.
```

Nothing is installed automatically, and there is no non-matplotlib fallback.
Install `matplotlib` into the environment you run the renderers from, then
re-run.  On machines where `$HOME/.config` is not writable, set `MPLCONFIGDIR`
to a writable directory to silence matplotlib's cache warning.  The headless
`Agg` backend is selected by the renderers themselves, so no display is needed.

## Tests

```bash
cd makeup
python -m unittest src.studies.paper_figures.tests.test_render_paper_figures -v
```

The suite builds synthetic metric JSONs in a temporary directory.  The
validation tests (missing file / missing key / non-numeric score / CLI exit
code) always run; the three PDF-writing tests are skipped with an explicit
reason when `matplotlib` is unavailable, because no PDF can be produced without
it.
