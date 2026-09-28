# CLEVRER Relation Controls

> **迁移后入口(note)**:下面的命令来自迁移前的源仓库,其中 `recipe.*` 模块路径
> 在本仓库已不存在,仅作历史记录保留。当前用法以
> `scripts/<dataset>/<experiment>/<task>.sh`(`analyses` 为
> `scripts/analyses/<analysis>/<task>.sh`)为准,完整顺序见 `docs/experiment.md`。


This recipe evaluates frozen CLEVRER decoder representations with pair-level
linear readouts. It compares `Full`, `Entity+Dynamic`, `Entity+Relation`,
`Dynamic+Relation`, `Relation-only`, `Entity`, and `Dynamic` without training a
new world model or structured probe.

Example:

```bash
PYTHONPATH="$PWD" python -m \
  recipe.recipe_formal.clevrer_analyseRelationControls.run \
  --train-latents /path/to/latents/train \
  --validation-latents /path/to/latents/validation \
  --train-targets /path/to/targets_train.pt \
  --validation-targets /path/to/targets_validation.pt \
  --checkpoint /path/to/structured_probe/best.pt \
  --output /tmp/clevrer_relation_controls.json
```

The readouts are fitted on train scenes and evaluated on validation scenes.
Each row is one valid object pair. Relation features are pooled from the
frozen `pair_time_features`; no future RGB or validation labels are used for
fitting.

The recommended background launcher uses eight GPUs for feature extraction:

```bash
bash recipe/recipe_formal/clevrer_analyseRelationControls/scripts/run.sh \
  --max-samples 1000 --output-mode workspace
```

Remove `--max-samples 1000` for the full run. Logs are written to the output
directory; `export.log` contains per-rank extraction progress and
`analysis.log` contains the merged readout metrics.
