# V-JEPA2 Physion++ full-patch dynamics decoder v1

> **迁移后入口(note)**:下面的命令来自迁移前的源仓库,其中 `recipe.*` 模块路径
> 在本仓库已不存在,仅作历史记录保留。当前用法以
> `scripts/<dataset>/<experiment>/<task>.sh`(`analyses` 为
> `scripts/analyses/<analysis>/<task>.sh`)为准,完整顺序见 `docs/experiment.md`。


This recipe is the Physion++ counterpart of the CLEVRER structured decoder.
It freezes the V-JEPA encoder and native Physion++ predictor, trains a shared
`UniversalDynamicsDecoder` plus a Physion++ structured probe, and supervises
segmentation-derived 2D object boxes, 2D motion, pair distance, collision,
first-contact, and scene-level OCP.

The implementation is isolated under `scripts/physionpp/`; the copied
`scripts/clevrer/` files are historical references and are not used by this
recipe.

## Smoke test and training

```bash
python -m recipe.vjepa2_naive_probe_v5_decoder_physionpp2.scripts.physionpp.prepare_targets \
  --dataset-root /data/ABDUCTIVE-WORLD/physion_v2/extracted \
  --split data_v1 --output /tmp/physionpp2_targets.pt --max-videos 2

bash recipe/vjepa2_naive_probe_v5_decoder_physionpp2/scripts/physionpp/run_all.sh \
  --output-mode workspace --dry-run
```

`run_all.sh` performs target generation, frozen encoder/predictor latent
caching, and 8-GPU probe training in the background. The default predictor is
`outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/best.pt`.

GT segmentation boxes are generated from paired `_id.json` RLE masks, not by
thresholding compressed RGB frames. The probe visualization uses H.264 output.

This recipe trains a `decoder_0810`-style universal dynamics decoder. It
produces `z_dyn [B,8,8,256]`; CLEVRER shallow probes read only `z_dyn` and use
the existing object, trajectory, distance, contact, and first-contact labels.

Each 128-frame video is exported as three non-overlapping samples by default:
window starts `0,32,64`. Their current/future ranges are `0..30 → 32..62`,
`32..62 → 64..94`, and `64..94 → 96..126` with raw frame step 2. Override
`WINDOW_STARTS` when a different window policy is required.

This experiment freezes the existing naive CLEVRER world model, caches its
complete `16x16` patch grid, trains an object-centric structured probe, and
exports the supervised object, object-time, pair, and pair-time predictions into
the shared CLEVRER v2 QA pipeline.

The detailed architecture, target definitions, losses, and QA integration are
documented in [`docs/probe_architecture.md`](docs/probe_architecture.md).

World-model checkpoint:

```text
/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/
clevrer_vith_16to16_stride2_8gpu/best.pt
```

The probe receives context and predicted-future tensors with shape
`[B,8,256,1280]` each. Six object queries attend to all 4096 context/future
patch tokens. Hungarian matching removes CLEVRER object-id ordering ambiguity;
trajectory validity, object presence, attributes, pair validity, contact, and
first-contact/no-contact are supervised explicitly.

## Output modes

Every public stage accepts `--output-mode both|data|workspace`; the default is
`both`.

- `both`: large caches/checkpoints go to `/data`; workspace keeps
  logs and small JSON/manifest files.
- `data`: complete output goes only to `/data`.
- `workspace` (default): complete output goes only to `outputs/runs/`.

Inspect paths without starting work:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/run_all.sh \
  --output-mode workspace --dry-run
```

Run target generation, full-patch cache export, and structured-probe training
sequentially in one background job:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/run_all.sh \
  --output-mode workspace
```

The full cache is expected to require roughly 157 GB for train+validation.
At the time this recipe was added, `/data` had only about 101 GB free, so
`workspace` is the runnable choice until `/data` has enough capacity. The cache
stage uses 8 GPUs by default and is resumable:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/cache_full_latents.sh \
  --output-mode workspace --resume
```

After `clevrer_structured_probe_seed239_stable_v2/best.pt` exists, inspect and start QA:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/qa_eval_8gpu.sh \
  --output-mode workspace --dry-run

bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/qa_eval_8gpu.sh \
  --output-mode workspace
```

The QA readout sequence is `CLS + 256 current/future visual tokens + 357
structured probe tokens + question/choice tokens`. Its run and evaluation names
are `clevrer_vith_16to16_v2_structured_supervised_route_8gpu` and
`qa_structured_supervised_route`, so it does not overwrite the earlier gated
structured readout.

For a minimal smoke test, use a new temporary output root and small scene
limits, for example `MAX_SCENES=8`, `MAX_TRAIN_SCENES=8`,
`MAX_VALIDATION_SCENES=8`, `EPOCHS=1`, and `NUM_GPUS=1`.

## Probe evaluation

### Single-clip video intervention

对一个固定 seed 随机选取的 clip，可以直接比较原视频、纯色背景替换、水平翻转、垂直翻转，以及从另一个随机 donor clip 合成一个物体后的 `add_object` 版本。该流程会重新运行 frozen encoder、native predictor 和 shallow probe，并保存完整 probe tensor、变换后的 expected targets 以及相对 `base` 的差异：

```bash
conda activate vjepa2-312
PYTHONPATH="$PWD" python -m \
  recipe.vjepa2_naive_probe_v5_decoder_physionpp3.scripts.physionpp.evaluate_video_intervention \
  --dataset-root /data/ABDUCTIVE-WORLD/physion_v2/extracted \
  --split readout_data_v1 \
  --seed 239 \
  --config recipe/vjepa2_naive_probe_v5_decoder_physionpp3/configs/physionpp-fullpatch-structured-probe-v1-8gpu.yaml \
  --predictor-checkpoint outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/best.pt \
  --probe-checkpoint <structured_probe>/best.pt \
  --output-dir outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/intervention_single_clip_seed239
```

输出的 `comparison.json` 中，背景替换应主要检查不变性；翻转实验除普通差异外，还会报告将 `base` 轨迹按几何规则变换后的等变性误差；`metadata.json` 会记录 donor 视频、donor object 和类别，`ground_truth.json` 会将新增物体加入 expected targets。脚本只做 shallow probe 评测，不修改训练代码或 checkpoint。

### 全测试集批量评测

批量入口默认遍历 `testdata_v1` 全部视频。模型只加载一次，逐视频执行五种版本并将逐视频结果写入 `per_video.jsonl`，最后生成 `aggregate.json` 和便于阅读的 `summary.md`。为控制磁盘占用，批量入口默认不保存每个视频的 mp4 和原始 probe tensor：

正式全量运行使用 8 张 GPU（按视频分片，每个 rank 独立加载一次 frozen encoder/predictor/probe）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29652 \
  -m recipe.vjepa2_naive_probe_v5_decoder_physionpp3.scripts.physionpp.evaluate_video_interventions_all \
  --dataset-root /data/ABDUCTIVE-WORLD/physion_v2/extracted \
  --split testdata_v1 --seed 239 \
  --config recipe/vjepa2_naive_probe_v5_decoder_physionpp3/configs/physionpp-fullpatch-structured-probe-v1-8gpu.yaml \
  --predictor-checkpoint outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/best.pt \
  --probe-checkpoint outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/physionpp3_metadata_probe_seed239_v1/structured_probe/best.pt \
  --output-dir outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/intervention_all_test_seed239
```

```bash
conda activate vjepa2-312
PYTHONPATH="$PWD" python -m \
  recipe.vjepa2_naive_probe_v5_decoder_physionpp3.scripts.physionpp.evaluate_video_interventions_all \
  --dataset-root /data/ABDUCTIVE-WORLD/physion_v2/extracted \
  --split testdata_v1 \
  --seed 239 \
  --config recipe/vjepa2_naive_probe_v5_decoder_physionpp3/configs/physionpp-fullpatch-structured-probe-v1-8gpu.yaml \
  --predictor-checkpoint outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/best.pt \
  --probe-checkpoint outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/physionpp3_metadata_probe_seed239_v1/structured_probe/best.pt \
  --output-dir outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/intervention_all_test_seed239
```

先用 `--max-videos 4` 做 smoke test；中断后可使用 `--resume` 继续已完成的视频。批量汇总采用逐视频 macro-average，失败视频会记录在 `per_video.jsonl`，不会静默丢弃。

All three launchers run in the background and support
`--output-mode both|data|workspace`. Evaluate matched object, trajectory, pair,
contact, and first-contact metrics on all 5,000 validation scenes:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/evaluate_structured_probe.sh \
  --output-mode workspace
```

Create a deterministic random scene report:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/report_structured_probe_sample.sh \
  --output-mode workspace
```

Create the corresponding report, source video, and H.264 overlay:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/visualize_structured_probe_sample.sh \
  --output-mode workspace
```

Set `SCENE_ID=10000` to inspect a chosen validation scene. Set `MAX_SCENES`
for a smaller numerical evaluation. Predictions are aligned once per scene
with the same assignment used during training; visualisation does not rematch
objects independently at each frame.
### OCP ShallowProbe 干预对照

在同一 `readout_data_v1` validation 样本上比较 visual-only、visual+probe、主要物体 mask、无关物体 mask 和随机 slot 注入。该评测支持 `torchrun` 8 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29653 \
  -m recipe.vjepa2_naive_probe_v5_decoder_physionpp3.scripts.physionpp.evaluate_ocp_interventions \
  --cache-root outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp2/physionpp_segmentation_2d_decoder_seed239_v2/cache \
  --targets outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/targets/targets_readout_data_v1.pt \
  --split readout_data_v1 --batch-size 8 \
  --probe-checkpoint outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/physionpp3_metadata_probe_seed239_v1/structured_probe/best.pt \
  --visual-checkpoint outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/physionpp3_metadata_probe_seed239_v1/ocp_visual_only/best.pt \
  --visual-probe-checkpoint outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/physionpp3_metadata_probe_seed239_v1/ocp_readout/best.pt \
  --output outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/ocp_interventions_validation_8gpu.json
```

先加 `--max-samples 16` 做 smoke test。输出 JSON 中的 `paired_delta_vs_visual_probe` 是三个 probe 干预相对完整 visual+probe baseline 的 OCP logit 变化。
