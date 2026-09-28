# AWM 主实验运行说明

本文说明如何在当前仓库运行论文正文的三个 AWM 主实验：Physion++、CLEVRER 和 EPIC-KITCHENS-100（EK100）。命令均从仓库根目录执行：

```bash
cd /home/shyang/workspace/work/awm_final
conda activate vjepa2-312
```

仓库不包含数据集和大模型权重。用户需要自行准备数据、V-JEPA 2 ViT-H encoder，以及 CLEVRER 所需的预测器 checkpoint。配置文件中的默认绝对路径只代表开发环境，发布运行前必须替换为本机路径。

## 通用约定

每个阶段都是一个 launcher：

```bash
bash scripts/<dataset>/<experiment>/<task>.sh [launcher options]
```

正式运行前先执行 `--dry-run`。launcher 默认后台运行，并在输出目录保存 command、resolved config、manifest、environment 和日志。

常用选项：

```text
--dry-run                         只解析配置和命令，不启动任务
--output-mode both|data|workspace 输出到 /data、workspace 或两者
--existing-output archive|raise   已有输出的处理方式
--background true|false           是否后台运行
--resume                          从已有 checkpoint 恢复
```

输出目录由配置中的 `launch.outputs` 决定：

```text
workspace: outputs/runs/<dataset>/<experiment>/<task>/
data:      /data/shyang/outputs/awm/<dataset>/<experiment>/<task>/
```

正式运行前检查 GPU、磁盘和端口：

```bash
nvidia-smi
df -h
ss -ltn
```

不要让多个任务使用同一个输出目录。每次新的正式运行都应使用新的实验名，或显式选择合适的 `--existing-output` 策略。

## 用户需要提供的资源

| 实验 | 数据 | 模型/权重 |
|---|---|---|
| Physion++ | extracted 数据，包含 `data_v1`、`readout_data_v1`、`testdata_v1`，视频及同 stem 的 `.pkl` 元数据 | 通用 V-JEPA 2 ViT-H checkpoint，默认 key 为 `target_encoder` |
| CLEVRER | 视频、processed proposals、官方 QA annotations | 通用 V-JEPA 2 ViT-H checkpoint；兼容配置的 CLEVRER stride-2 world-model checkpoint |
| EK100 | EPIC-KITCHENS-100 视频和官方 train/validation CSV | 通用 V-JEPA 2 ViT-H checkpoint，默认 key 为 `target_encoder` |

模型结构必须与配置中的 ViT-H 参数和 checkpoint key 兼容。仓库不会自动下载、转换或修复不兼容的 checkpoint。

## 配置自己的路径

不要直接修改仓库内的默认配置。复制后修改：

```bash
mkdir -p "$HOME/awm-configs"
cp configs/physionpp/awm_physionpp_fullpatch_probe_seed239/config.yaml "$HOME/awm-configs/physionpp.yaml"
cp configs/clevrer/awm_clevrer_fullpatch_probe_seed239/config.yaml "$HOME/awm-configs/clevrer.yaml"
cp configs/ek100/awm_ek100_multiscale_adapter_seed239/config.yaml "$HOME/awm-configs/ek100.yaml"
```

用 `CONFIG` 指定私有配置：

```bash
CONFIG="$HOME/awm-configs/physionpp.yaml" bash scripts/physionpp/awm_physionpp_fullpatch_probe_seed239/train_predictor.sh --dry-run
```

路径在同一配置的多个 task block 中可能重复出现，必须全部保持一致：Physion++ 的 `physion_root`、`encoder_checkpoint`、`data.root` 和 `meta.pretrain_checkpoint`；CLEVRER 的 `clevrer_root`、`clevrer_annotations`、`encoder_checkpoint`、`world_checkpoint` 及 `tasks.train.experiment` 中的 `data.root`；EK100 的 `ek100_root`、两个 annotation CSV、`encoder_checkpoint` 及 cache/readout/adapter task 中的对应字段。

## Physion++ 主实验

协议为 `data_v1 -> readout_data_v1 -> testdata_v1`：训练和选择只使用前两个 split，`testdata_v1` 只用于最终 OCP 评测。

```bash
EXP=scripts/physionpp/awm_physionpp_fullpatch_probe_seed239
bash "$EXP/train_predictor.sh" --dry-run
bash "$EXP/train_predictor.sh"
bash "$EXP/cache_train.sh"
bash "$EXP/cache_readout.sh"
bash "$EXP/cache_test.sh"
bash "$EXP/train_probe.sh"
bash "$EXP/eval_ocp.sh"
```

可选的校准评测：`bash "$EXP/eval_ocp_calibrated.sh"`。结果位于 `eval_ocp/`，包括 OCP AUROC、balanced accuracy 和 accuracy。该流程重新训练 native predictor、structured probe 和 OCP readout，不加载历史 V12/V20/V23 predictor checkpoint。

## CLEVRER 主实验

协议为 `prepare_targets -> cache_latents -> train_probe -> eval_probe -> qa_eval`：

```bash
EXP=scripts/clevrer/awm_clevrer_fullpatch_probe_seed239
bash "$EXP/prepare_targets_train.sh" --dry-run
bash "$EXP/prepare_targets_train.sh"
bash "$EXP/prepare_targets_validation.sh"
bash "$EXP/cache_latents_train.sh"
bash "$EXP/cache_latents_validation.sh"
bash "$EXP/train_probe.sh"
bash "$EXP/eval_probe.sh"
```

确认 `train_probe` 生成 `best.pt` 后运行：

```bash
bash "$EXP/qa_eval.sh" --dry-run
bash "$EXP/qa_eval.sh"
```

QA 使用官方 validation 标注；官方 test 无标签，只能生成 submission，不能报告 test accuracy。完整 latent cache 可能需要约 157 GB。仅有原始视频而没有 processed proposals 时，不能直接运行该主实验。`world_checkpoint` 必须是兼容的 CLEVRER 预测器，不是通用 ViT-H encoder。

## EK100 主实验

协议为 `train annotations -> video-disjoint indices -> frozen encoder cache -> multiscale readout -> shared temporal adapter -> official validation evaluation`：

```bash
EXP=scripts/ek100/awm_ek100_multiscale_adapter_seed239
bash "$EXP/prepare_indices_train.sh" --dry-run
bash "$EXP/prepare_indices_train.sh"
bash "$EXP/prepare_indices_validation.sh"
bash "$EXP/cache_train.sh"
bash "$EXP/cache_probe_validation.sh"
bash "$EXP/cache_final_validation.sh"
bash "$EXP/train_readout.sh"
bash "$EXP/train_adapter.sh"
bash "$EXP/evaluate.sh"
```

readout 和 adapter 只使用 train 以及从 train 划出的 probe-validation；`final_validation` 仅用于最终报告。结果位于 `evaluate/metrics.json`，包括 verb、noun 和 action 的 Top-1/Top-5。

## 复现记录和限制

每个任务完成后检查 `command.txt`、`manifest.json`、`environment.json`、`configs/<task>.resolved.yaml` 和 `logs/<task>.log`。这些入口是三条可运行的 AWM task pipeline，不保证任意环境能够直接复现论文历史数字；未完成完整训练和评测前，不应把论文已有数字写成新运行结果。完整 launcher 契约见 [`tools/launcher/README.md`](../../tools/launcher/README.md)。
