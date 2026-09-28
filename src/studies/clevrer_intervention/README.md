# CLEVRER intervention analysis

> **迁移后入口(note)**:下面的命令来自迁移前的源仓库,其中 `recipe.*` 模块路径
> 在本仓库已不存在,仅作历史记录保留。当前用法以
> `scripts/<dataset>/<experiment>/<task>.sh`(`analyses` 为
> `scripts/analyses/<analysis>/<task>.sh`)为准,完整顺序见 `docs/experiment.md`。


Run one frozen-model predictive QA example:

```bash
conda activate vjepa2-312
PYTHONPATH="$PWD" python -m recipe.recipe_formal.clevrer_analyseIntervention.random_predictive_qa \
  --seed 239 \
  --output-root outputs/runs/reproduce_v1_clevrer_only/clevrer_analyseIntervention/random_predictive_qa
```

The command samples one predictive question from official validation
annotations and uses the formal QA trajectory (`scene_XXXXX.pt`) containing
observed latent steps for raw frames `96,100,...,124` and predictor-imagined
steps for raw frames `128,132,...,156`. It runs the frozen QA checkpoint and
writes the question, choices, probabilities and predicted answer to JSON. The
`video` field points to the corresponding original validation video; the video
is copied into the same seed-specific output directory as the JSON result.

默认目录结构为：

```text
outputs/runs/reproduce_v1_clevrer_only/clevrer_analyseIntervention/random_predictive_qa/
└── seed_239/
    ├── random_predictive_qa.json
    └── video_XXXXX.mp4
```

The old full-patch `window_start=64` cache is intentionally not used because
it is a different multi-window protocol. If the trajectory directory is
absent, first run the repository QA trajectory exporter with the formal QA
config and structured probe enabled, writing to `--trajectory-root`.

一次性导出全部 CLEVRER split 的 current/future latent：

```bash
conda activate vjepa2-312
PYTHONPATH="$PWD" python -m \
  recipe.recipe_formal.clevrer_analyseIntervention.export_predictive_latents \
  --split all --device cuda:0
```

默认写入 `/data/shyang/outputs/vjepa2-baiz/reproduce_v1_clevrer_only/`
下的 `clevrer_predictive_latents_v1/{train,validation,test}/`，每个 scene 一个
`scene_XXXXX.pt`。可以加 `--resume` 跳过已有文件；正式运行建议用
`torch.distributed.run --nproc_per_node 8`。
其中 `observed_tokens` 和 `future_tokens` 均为 `[8,256,1280]`，并保存拼接后的
`visual_tokens [16,256,1280]`。

对 validation 全部 predictive questions 运行 probe mask 对照实验：

```bash
PYTHONPATH="$PWD" torchrun --nproc_per_node=8 -m \
  recipe.recipe_formal.clevrer_analyseIntervention.eval_validation_probe_mask_qa \
  --seed 239 \
  --device cuda:0
```

默认结果写入：

```text
outputs/runs/reproduce_v1_clevrer_only/clevrer_analyseIntervention/
probe_mask_qa_validation/validation_probe_mask_qa.json
```

JSON 的 `metrics` 字段分别报告 `baseline`、`relevant_mask`、
`irrelevant_mask` 的 `question_accuracy` 和 `option_accuracy`；`records` 保存
每个问题的 object slot 选择和三组预测结果。可先用 `--max-scenes 3` 做 smoke test。
