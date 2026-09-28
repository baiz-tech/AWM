# resource.md

本文件记录**当前机器**的真实资源信息。`requirement.md` §4 要求机器专用路径、
IP、网卡、端口不进入通用源码;需要这些值的 config 通过本文件核对,或通过环境变量覆盖。

> 本文件描述的是记录时刻的实测状态。机器变化后请同步更新,不要凭记忆沿用。

## 1. 节点

| 项 | 值 |
|---|---|
| hostname | `HOST-10-118-7-233` |
| 操作系统 | Rocky Linux 9.2 (Blue Onyx) |
| kernel | `5.14.0-284.25.1.el9_2.x86_64` |
| CPU | 255 cores |
| 内存 | 1003 GB |
| `/data` 文件系统 | 3.5 TB,已用 2.7 TB,可用 643 GB(81%) |

## 2. GPU

| 项 | 值 |
|---|---|
| 期望配置 | 8 × NVIDIA A100 80GB |
| **记录时刻实测状态** | **不可用**:`nvidia-smi` 无法与驱动通信,`/dev/nvidia*` 不存在 |

因此本次代码整理**只完成了 CPU 级验证**,所有 GPU 端到端 smoke test 均未执行。

## 3. Python 环境

```text
解释器: /data/shared/envs/vjepa2-312/bin/python
Python: 3.12.13 (CPython)
torch:  2.6.0+cu124
torchvision: 0.21.0+cu124
numpy:  2.5.1
```

已确认可导入的依赖:`yaml 6.0.3`、`decord 0.6.0`、`cv2 5.0.0`、`scipy 1.18.0`、
`timm 1.0.27`、`pandas 3.0.3`、`einops 0.8.2`、`PIL 12.3.0`。

未安装:`pytest`(因此测试用 `unittest` 运行,见根目录 `README.md`)。

## 4. 数据与权重

### 4.1 本机可用

| 资源 | 路径 | 说明 |
|---|---|---|
| Physion++ 数据 | `/data/ABDUCTIVE-WORLD/physion_v2/extracted` | 含 `data_v1`、`readout_data_v1`、`testdata_v1`、`humans` |
| V-JEPA 2 ViT-H 权重 | `/data/ABDUCTIVE-WORLD/pretrain/checkpoints/vith.pt` | 10.4 GB,key 为 `target_encoder` |
| CLEVRER 原始数据 | `/data/shared/datasets/CLEVRER` | 51 GB,含 `videos/{train,val,test}` 与 `official_code/executor/data` |
| CLEVRER naive world model | `/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/clevrer_vith_16to16_stride2_8gpu/best.pt` | 256 MB,V-JEPA 2 matched baseline |

### 4.2 本机缺失

| 资源 | 期望路径 | 影响 |
|---|---|---|
| EK100 数据 | `/data/shared/datasets/EPIC-KITCHENS-processed_v1` | 所有 EK100 实验与 VideoMAE2-ek100 无法在本机运行 |
| Orca-4B 权重 | `/data/shared_model/Orca-4B` | 两个 Orca baseline 无法在本机运行 |
| 历史 run 输出 | `/data/shyang/outputs/ABDUCTIVE-WORLD/**` | paper 中引用的 cache / probe / readout 均不存在,离线分析无法复跑 |
| CLEVRER latent cache | `/data/shyang/outputs/vjepa2-baiz/reproduce_v1_clevrer_only/**` | 同上 |

### 4.3 体积提示

CLEVRER 全量 latent cache 约需 157 GB。`/data` 当前可用 643 GB。整理本仓库时
**没有把任何 checkpoint、cache 或数据集副本放入 `makeup/`**。

## 5. 端口

单机 8 卡默认使用 `launch.resources.master_port`。各实验 config 当前统一使用
`29500`;历史上 CLEVRER QA 流水线使用 `29720` 作为基准端口并按阶段递增
(`+0/+3/...`)。多节点时通过 `MASTER_PORT` 环境变量覆盖。

## 6. 环境变量覆盖约定

`tools/launcher/launch_from_config.sh` 允许以下环境变量覆盖 config 默认值:

```text
NNODES
NODE_RANK
MASTER_ADDR
MASTER_PORT
NPROC_PER_NODE
CUDA_VISIBLE_DEVICES
```

输出位置由 `--output-mode both|data|workspace` 控制;`data_root` 默认
`/data/shyang/outputs`,workspace 侧默认 `outputs/runs`。
