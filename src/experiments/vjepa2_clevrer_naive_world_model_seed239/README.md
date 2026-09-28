# V-JEPA 2 Native Predictor Recipe

> **迁移后入口(note)**:下面的命令来自迁移前的源仓库,其中 `recipe.*` 模块路径
> 在本仓库已不存在,仅作历史记录保留。当前用法以
> `scripts/<dataset>/<experiment>/<task>.sh`(`analyses` 为
> `scripts/analyses/<analysis>/<task>.sh`)为准,完整顺序见 `docs/experiment.md`。


本目录实现 Physion++ 上的 V-JEPA 2 原生 predictor 公平 baseline：

```text
current 16 sampled frames → predict future 16 sampled-frame latent tokens
```

实现完全隔离在 `recipe/vjepa2_naive/`，不要求修改仓库其他代码。

## 快速启动

```bash
cd /home/shyang/workspace/work/vjepa2-baiz
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/train_4gpu.sh --dry-run
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/train_8gpu.sh
bash recipe/vjepa2_naive/scripts/physionpp/gap32/train_8gpu.sh
bash recipe/vjepa2_naive/scripts/clevrer/stride2/train_8gpu.sh --dry-run
```

新配置按数据集分组：Physion++ 位于 `configs/physionpp/`，CLEVRER 位于
`configs/clevrer/`。训练入口统一为 `recipe.vjepa2_naive.train`，数据集类型和采样协议
由 config 中的 `experiment.data.dataset`、frame step、`clip_gap` 和
`experiment.protocol` 字段控制。
CLEVRER 默认只报告训练期 one-step latent validation 的 loss/MSE/cosine，不再包含
two-step closed-loop test。

每个 YAML 顶层只分为 `launch` 和 `tasks`。`launch` 定义 run 名称、默认输出目录和
默认 GPU/端口；`tasks.<task>.launch` 定义该任务怎么启动，`tasks.<task>.experiment`
定义数据、模型、优化、训练日志和 QA/OCP 等真正影响实验结果的参数。

训练结束后的 OCP 和时序检索入口位于同一设置目录：
`recipe/vjepa2_naive/scripts/physionpp/no_gap/`。

在 `testdata_v1` 上比较单步预测 latent 与真实 Future teacher latent：

```bash
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/eval_future_prediction_8gpu.sh --dry-run
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/eval_future_prediction_8gpu.sh
```

该 8GPU 后台任务输出 `evaluations/future_prediction_test/metrics.json` 和逐视频
`per_video_metrics.jsonl`，报告 prediction 与 current-copy baseline 的
MAE、MSE、RMSE、cosine similarity/distance 和 relative L2。

单 Future clip 接触评测固定使用 8 张 GPU：

```bash
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/eval_single_future_ocp_8gpu.sh --dry-run
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/eval_single_future_ocp_8gpu.sh
```

该协议只预测一个 Future clip。标签检查首个与末个 Future 采样帧之间的每一张原始帧；
例如采样帧为 `[P,P+4,...,P+60]` 时，标签区间为 `[P,P+60]`，因此中间未采样帧的
`target_contacting_zone` 也会计入。Future RGB 不输入模型，结果写入
`evaluations/single_future_clip_ocp/`。

只预测同一个 Future clip、但使用整个 Future 接触标签的评测同样固定使用 8 张 GPU：

```bash
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/eval_single_future_all_future_ocp_8gpu.sh --dry-run
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/eval_single_future_all_future_ocp_8gpu.sh
```

该协议仍只预测 `[P,P+4,...,P+60]`，但标签检查 `[P,video_end)` 的每一张原始帧。
结果写入 `evaluations/single_future_all_future_ocp/`，不会覆盖单 clip 标签结果。

## 文档导航

- [方法介绍](docs/introduction.md)
- [模型与训练架构](docs/architecture.md)
- [数据采样与 16→16 协议](docs/data_protocol.md)
- [安装、训练、恢复和输出说明](docs/usage.md)

## 代码导航

| 文件 | 作用 |
|---|---|
| `dataset.py` | 抽取不重叠的 current/future 16 帧 clip |
| `model.py` | 冻结 encoder、原生 predictor 和前后半区 mask |
| `train.py` | 单卡/DDP 训练、验证和 checkpoint；评测实现位于 `recipe/shared/evaluate/` |
| `configs/physionpp/`, `configs/clevrer/` | 按数据集分组的训练与评测配置 |
| `scripts/physionpp/` | Physion++ 的训练与评测入口 |
| `scripts/clevrer/` | CLEVRER 的训练与 QA probe 入口 |

数据集分组脚本位于 `scripts/physionpp/` 和 `scripts/clevrer/`。评测实现代码位于
`recipe/shared/evaluate/physionpp/` 和 `recipe/shared/evaluate/clevrer/`，供不同实验复用；
评测 shell 入口仍放在当前实验的 `scripts/` 目录中。CLEVRER 仅保留 QA probe 入口，
不迁移 two-step test。

默认只用 `data_v1` 训练 predictor，用 `readout_data_v1` 验证 latent prediction；`testdata_v1` 不参与训练和调参。

## OCP readout 与最终测试

predictor 训练结束后运行：

```bash
python -m recipe.shared.evaluate.physionpp.eval_full_future_ocp \
  --eval-config recipe/vjepa2_naive/configs/physionpp/physion-vith-16to16-full-future-ocp.yaml
```

也可以使用后台启动脚本（参数可在上述 YAML 中修改）：

```bash
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/eval_full_future_ocp.sh
```

训练和评测脚本只声明 `CONFIG` 与 `TASK`，再通过
`recipe/shared/tools/launcher/launch_from_config.sh` 从 config 解析运行设置，并由底层
`recipe/shared/tools/launcher/launcher_common.sh` 后台启动实际 Python/torchrun 进程。
日志、PID 和产物同时写入 `/data/shyang/outputs/...` 与
workspace mirror；评测产物写入对应 run 的 `evaluations/<eval_name>/`。新任务遇到已有输出目录时，
会先将旧目录归档到同级 `legacy/`；使用 `--resume` 恢复训练时保留原目录。
训练按最低 validation prediction loss 保存 `best.pt`，两个评估阶段默认使用
该 checkpoint；`latest.pt` 仍用于恢复最近训练状态。

默认统一协议将 `readout_data_v1` 按 path hash 和类别固定拆成 600 个 probe-fit 与
200 个 threshold-validation 样本，随后冻结探针和阈值，在 `testdata_v1` 上一次性报告
`full/current/rollout_mean/last_chunk/delta` 的 accuracy、balanced accuracy、F1 和 AUROC。
测试协议与 `vjepa2_ef4` unified probe 对齐：每个视频从真实 Current 开始，按实际
剩余长度 closed-loop rollout 到视频末尾（不解码或读取真实 Future）；特征为
Current mean、全 Future prediction mean、最后一个 chunk mean 及 delta 的拼接，
标签为 `[P+clip_gap, video_end)` 中是否曾发生 contact。`naive` checkpoint 没有
multi-Future 的可训练 rollout adapter，因此直接把前一预测 latent 作为下一步
context，并在 `metrics.json` 中显式记录这一点。
输出默认写入运行目录下的 `evaluations/ocp/`：`metrics.json`、
`test_predictions.npz` 和可复用的 `readout.pt`。

若配置了 `evaluation.humans_dir`，评估还会读取其中所有
`human_accuracy-*.csv`：`human_all_accuracy` 在全部匹配的人类实验 stimulus
上计算，`human_hard_accuracy` 默认在人类加权正确率不高于 0.5 的 stimulus
上计算。跨 run 的人类正确率按每行响应人数 `c` 加权合并；最终模型正确性
以 human CSV 的 benchmark label 为准。
