# AWM 主实验运行说明

本文说明如何在当前仓库运行论文正文的三个 AWM 主实验：Physion++、CLEVRER 和 EPIC-KITCHENS-100（EK100）。命令均从仓库根目录执行：

## 配置运行环境

使用 Linux、Python 3.12 和支持 CUDA 的 NVIDIA GPU 环境。先安装 Miniconda 或 Anaconda，并确保 `nvidia-smi` 能正常显示 GPU。以下安装示例采用参考环境的 PyTorch 2.6.0、torchvision 0.21.0 和 CUDA 12.4 wheel；显卡驱动须支持该 CUDA 运行时。

从仓库根目录创建独立环境并安装依赖：

```bash
conda create -n awm python=3.12 pip -y
conda activate awm
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

安装完成后检查依赖和 GPU：

```bash
python -m pip check
python - <<'PY'
import torch
import torchvision
import yaml, decord, cv2, scipy, timm, numpy

print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA 不可用，请检查 NVIDIA 驱动和 PyTorch 安装"
PY
```

每次运行实验前执行 `conda activate awm`，使 `python` 和 `torchrun` 使用同一环境。仓库默认配置按单机 8 卡运行；单卡检查需要同步修改 GPU 数和可见设备。若配置中的 `python_bin` 使用了开发机器的绝对路径，应替换为当前环境的解释器路径（`command -v python`）。`requirements.txt` 中多数依赖使用版本下限，以上步骤并未锁定全部依赖版本；运行前仍需完成 dry-run 和短程检查。

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

## 数据集、模型和输出路径在哪里指定

三条主实验的默认配置分别位于：

```text
configs/physionpp/awm_physionpp_fullpatch_probe_seed239/config.yaml
configs/clevrer/awm_clevrer_fullpatch_probe_seed239/config.yaml
configs/ek100/awm_ek100_multiscale_adapter_seed239/config.yaml
```

建议复制配置到 `temp/` 或用户自己的配置目录后修改，不要直接修改仓库中的默认配置。路径字段的位置如下：

| 实验 | 数据集路径 | 模型路径 | 其他必须配置的路径 |
|---|---|---|---|
| Physion++ | `launch.extra_config.paths.physion_root` | `launch.extra_config.paths.encoder_checkpoint` | `launch.run.experiment`；各 task 中的 `data.root` 和 `meta.pretrain_checkpoint` |
| CLEVRER | `launch.extra_config.paths.clevrer_root` | `encoder_checkpoint` 和 `world_checkpoint` | `launch.extra_config.paths.clevrer_annotations`；`tasks.train.experiment.data.root` 和 `meta.pretrain_checkpoint` |
| EK100 | `launch.extra_config.paths.ek100_root`、`train_annotations`、`validation_annotations` | `launch.extra_config.paths.encoder_checkpoint` | `launch.run.experiment`；cache/readout/adapter task 中的 `data` 和 `weights` |

其中：

- `encoder_checkpoint` 是通用 V-JEPA 2 ViT-H encoder，通常使用 `target_encoder`，由 `encoder_checkpoint_key` 指定 key。
- CLEVRER 的 `world_checkpoint` 是已经训练好的 CLEVRER latent predictor，不能用通用 encoder checkpoint 替代。
- `physion_root`、`clevrer_root` 和 `ek100_root` 应指向数据集根目录，而不是某一个具体视频文件。
- EK100 的 `train_annotations` 和 `validation_annotations` 应指向官方 CSV 文件。
- 输出位置由 `launch.run.workspace_root`、`launch.run.data_root` 和 `launch.outputs` 指定；通常只需要通过 `--output-mode workspace|data|both` 选择输出模式。

例如，使用 `temp/physionpp-test.yaml` 时，修改以下字段：

```yaml
launch:
  run:
    experiment: physionpp_test_20260928
  extra_config:
    paths:
      physion_root: /absolute/path/to/physion_v2/extracted
      encoder_checkpoint: /absolute/path/to/vith.pt
      encoder_checkpoint_key: target_encoder
```

Physion++ 配置中同一条路径会在 predictor、cache 和 probe 的 task 配置中重复出现。修改后请搜索配置确认没有残留旧路径：

```bash
rg -n '/data/|physion_root|encoder_checkpoint|data.root|pretrain_checkpoint' temp/physionpp-test.yaml
```

运行时通过 `CONFIG` 选择该配置：

```bash
CONFIG="$PWD/temp/physionpp-test.yaml" \
  bash scripts/physionpp/awm_physionpp_fullpatch_probe_seed239/train_predictor.sh \
  --output-mode workspace --dry-run
```

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
