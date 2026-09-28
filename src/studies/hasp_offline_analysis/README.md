# HASP offline analysis

> **迁移后入口(note)**:下面的命令来自迁移前的源仓库,其中 `recipe.*` 模块路径
> 在本仓库已不存在,仅作历史记录保留。当前用法以
> `scripts/<dataset>/<experiment>/<task>.sh`(`analyses` 为
> `scripts/analyses/<analysis>/<task>.sh`)为准,完整顺序见 `docs/experiment.md`。


This directory contains analysis-only code. It never modifies training code or
checkpoints and reuses the existing frozen caches.

## 1. Export states

```bash
python -m recipe_formal.analyze.extract_physion \
  --cache /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v12_physion_only/seed239_parallel_v12/cache/validation \
  --checkpoint /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v12_physion_only/seed239_parallel_v12/probe/best.pt \
  --output /data/shyang/outputs/ABDUCTIVE-WORLD/analysis/physion_validation

python -m recipe_formal.analyze.extract_ek100 \
  --cache /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v23_ek100_only/seed239_parallel_v23/cache/probe_validation \
  --readout /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v23_ek100_only/seed239_parallel_v23/readout/best.pt \
  --adapter /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v23_ek100_only/seed239_parallel_v23/adapter/best.pt \
  --output /data/shyang/outputs/ABDUCTIVE-WORLD/analysis/ek100_probe_validation
```

Use `--max-samples 16 --device cpu` for a smoke test. The exported vectors are
compact pooled summaries of entity, dynamic, relation and full states.

## 2. Decode information

```bash
python -m recipe_formal.analyze.probe_matrix --features ... --task ocp --output results/physion_ocp.json
python -m recipe_formal.analyze.probe_matrix --features ... --task verb --output results/ek100_verb.json
```

The first version provides the layer × factor matrix. Subsequent analyses can
reuse the same per-sample files for state swaps, temporal interventions, and
hidden-cause recovery without running the encoder again.

## 3. Native Physion++ factor matrix

This is the primary functionality test. It uses physical annotations directly,
not OCP as a proxy:

```bash
python -m recipe_formal.analyze.physion_factor_probe \
  --train /path/to/physion_train_features \
  --test /path/to/physion_validation_features \
  --output results/physion_native_factor_matrix.json
```

Expected correspondence is entity → count/position/extent, dynamic → speed/
displacement/TTC, and relation → distance/contact/contact-rate.

For the stronger slot/time-resolved analysis:

```bash
python -m recipe_formal.analyze.physion_slot_time_probe \
  --train /path/to/physion_train_features \
  --test /path/to/physion_validation_features \
  --output results/physion_slot_time_matrix.json
```

This preserves slot identity through Hungarian matching and evaluates Entity
and Dynamic at their native granularity.

For pair-level relation analysis:

```bash
python -m recipe_formal.analyze.physion_pair_probe \
  --train /path/to/physion_train_features \
  --test /path/to/physion_validation_features \
  --output results/physion_pair_relation_matrix.json
```

It Hungarian-aligns slots first, then decodes distance, contact, and
time-to-contact for every valid object pair.

## 4. Physion++ Dynamic selectivity

`physion_dynamic_selectivity.py` tests whether the Dynamic state is selective
to state change and temporal order. It decodes native speed, acceleration and
TTC targets from the physical annotations using `entity`, `dynamic`,
`relation`, and `full` features. Optionally, it reruns the frozen structured
probe on cached-latent interventions (`future_reverse`, `future_shuffle`,
`future_static`, and `future_repeat`) and reports paired errors and deltas.

Baseline feature decoding:

```bash
python -m recipe.recipe_formal.analyze.physion_dynamic_selectivity \
  --train-features /path/to/physion_train_features \
  --test-features /path/to/physion_validation_features \
  --output results/physion_dynamic_selectivity.json
```

Add `--cache /path/to/cache --checkpoint /path/to/probe/best.pt` to run the
cached-future temporal interventions. `--max-samples` is available for a
small smoke test. Acceleration is computed from consecutive valid future
velocity states; TTC uses `future_time_to_contact` and its validity mask.

For the complete 8-GPU pipeline (sharded feature export, smoke test, full
analysis, logs and manifest), run:

```bash
bash recipe/recipe_formal/analyze/scripts/run_physion_dynamic_selectivity.sh \
  --run /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v12_physion_only/seed239_parallel_v12 \
  --output /data/shyang/outputs/vjepa2-baiz/analyses/physion_dynamic_selectivity_v1
```

Use `--dry-run` first. The launcher refuses to overwrite an existing output
directory; choose a new output path for a rerun.

## 5. CLEVRER Dynamic selectivity

CLEVRER exposes 2D velocity in `state[..., 5:7]`, pair contact, and
first-contact time bins. Run the offline analysis with:

```bash
bash recipe/recipe_formal/analyze/scripts/run_clevrer_dynamic_selectivity.sh \
  --data /data/shyang/outputs/vjepa2-baiz/reproduce_v1_clevrer_only/clevrer_fullpatch_data_v1 \
  --checkpoint outputs/runs/vjepa2_naive_probe_v5_decoder/clevrer_dynamics_decoder_multiwindow_seed239_retry_v1/best.pt \
  --output /data/shyang/outputs/vjepa2-baiz/analyses/clevrer_dynamic_selectivity_v1
```

The script compares Entity/Dynamic/Relation/Full features for speed,
acceleration, TTC (normalized first-contact bin), and contact event, then
evaluates future reverse/shuffle/static/repeat interventions.

## 5. Physion++ Relation instance selectivity

`physion_relation_instance_ablation.py` uses the aligned `*_id.mp4` instance
video to construct same-scene paired interventions. For each eligible sample it
selects a real future-contact pair, an area-matched non-contact pair, and an
area-matched random pair. The selected context patches are replaced and the
frozen predictor regenerates the future latent for every intervention.

All variants reuse the baseline Hungarian slot assignment. The output contains
global Entity/Dynamic/Relation/OCP metrics, pair-level contact AUROC, distance
MAE and TTC MAE, per-variant deltas, and pair-selection metadata.

```bash
bash recipe/recipe_formal/analyze/scripts/run_physion_relation_instance_ablation.sh \
  --run /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v12_physion_only/seed239_parallel_v12 \
  --output-mode workspace --max-samples 16 --dry-run
```

For a real run, omit `--dry-run`; the launcher runs in the background and
refuses to overwrite an existing output directory.

## 6. CLEVRER Relation instance selectivity

The CLEVRER counterpart uses proposal/instance masks from
`processed_proposals/sim_XXXXX.json` and pairs a future collision object pair
with same-scene non-collision and area-matched random pairs. Context patches are
masked, the frozen naive predictor regenerates future latents, and the frozen
structured decoder is evaluated on pair distance and contact.

```bash
bash recipe/recipe_formal/analyze/scripts/run_clevrer_relation_instance_ablation.sh \
  --run /data/shyang/outputs/vjepa2-baiz/reproduce_v1_clevrer_only/clevrer_fullpatch_data_v1 \
  --output-mode workspace --max-scenes 16 --dry-run
```

正式版默认遍历 validation target 中的全部 scene/window，并使用三个固定
random seeds（`239,241,251`）。`--area-tolerance` 默认 `0.25`，要求
non-contact/random pair 的 mask union 面积与 contact pair 的相对差异不超过
25%；不满足时该样本不会进入 eligible triplet，避免面积差异污染结论。
