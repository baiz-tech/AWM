# CLEVRER Interaction-Conditioned Future Prediction

> **迁移后入口(note)**:下面的命令来自迁移前的源仓库,其中 `recipe.*` 模块路径
> 在本仓库已不存在,仅作历史记录保留。当前用法以
> `scripts/<dataset>/<experiment>/<task>.sh`(`analyses` 为
> `scripts/analyses/<analysis>/<task>.sh`)为准,完整顺序见 `docs/experiment.md`。


This recipe evaluates the frozen structured probe by interaction regime. Each
valid object pair is assigned to `approaching`, `contact`, `separating`, or
`no-contact` from the ground-truth future trajectory, then distance, speed,
contact, and first-contact/TTC errors are reported per group.

```bash
PYTHONPATH="$PWD" python -m \
  recipe.recipe_formal.clevrer_analyseInteractionConditionedFuturePrediction.run \
  --validation-latents /path/to/latents/validation \
  --validation-targets /path/to/targets_validation.pt \
  --checkpoint /path/to/structured_probe/best.pt \
  --output /tmp/clevrer_interaction_conditioned.json
```

The decoder and probe are frozen. The latent cache is reusable when the
world-model checkpoint and sampling protocol are unchanged.
