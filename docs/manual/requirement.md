# requirement

本文件用于约束 coding agent 在本项目中的代码修改、实验组织、launcher、输出和分布式运行。

配套文件：

| 文件 | 作用 |
|---|---|
| `requirement.md` | 定义必须遵守的开发与运行规范 |
| `example_config.yaml` | 提供标准 config 结构和字段示例 |
| `resource.md` | 记录当前机器、路径、节点、网卡、环境等真实资源信息 |

> 机器专用路径、IP、网卡、端口等不得硬编码到通用源码中。

---

## 1. 通用规则

| 规则 | 要求 |
|---|---|
| 修改前检查 | 先阅读相关代码、config、测试和文档，并执行 `git status --short` |
| 用户改动 | 不覆盖、不回退、不重写用户未提交修改 |
| 修改范围 | 只做当前任务需要的最小改动 |
| Git | 不执行 `commit`、`push`、PR，除非用户明确要求 |
| 文件安全 | 不删除用户数据、checkpoint、日志或实验结果 |
| 错误处理 | 不静默吞异常，不伪造默认值、日志、结果或验证状态 |
| 配置来源 | 路径、设备、端口、checkpoint、超参数等从 config / CLI / env / `resource.md` 获取 |

---

## 2. 项目目录结构

### 2.1 公共目录

```text
<repo>/
├── README.md
│   # 项目总览、安装和基本使用说明
│
├── CONTRIBUTING.md
│   # 开发、测试和协作约定
│
├── pyproject.toml
│   # Python 项目和依赖配置
│
├── requirements.txt
│   # 环境依赖
│
├── .env.example
│   # 环境变量模板，不写真实密钥或机器专用值
│
├── configs/
│   # 实验配置，按 <dataset>/<experiment> 组织
│
├── scripts/
│   # 实验启动脚本，与 configs 对应
│
├── outputs/
│   # workspace 侧实验输出，通常 gitignored
│
├── src/
│   ├── core/
│   │   # 与 dataset 无关的模型、loss、公共接口
│   ├── training/
│   │   # 训练循环、checkpoint、distributed runtime
│   ├── data/
│   │   └── <dataset>/
│   │       # dataset adapter / codec / protocol
│   ├── experiments/
│   │   # 实验与诊断实现
│   └── studies/
│       # 独立 study / research 实现
│
├── tests/
│   # 单元、集成、launcher、distributed 等测试
│
├── docs/
│   # 项目文档
│
├── external/
│   # 第三方源码；除非明确要求，一般不修改
│
├── tools/
│   └── launcher/
│       # 公共 launcher
│
├── .runtime/
│   # 本机临时运行状态，gitignored
│
└── temp/
    # 临时文件；正式代码不得依赖
```

核心约束：

- `<dataset>`：实验使用的数据集。
- `<experiment>`：config、script、output 的共同主键。
- `tools/launcher/` 只放公共启动逻辑，不放具体实验业务。
- 历史 config / script / output 放入对应 `legacy/`。

### 2.2 单阶段 experiment

```text
configs/<dataset>/<experiment>/
└── config.yaml
    # 唯一 config，launch.run.stage = 0

scripts/<dataset>/<experiment>/
├── train.sh
├── eval.sh
└── <task>.sh
    # 薄启动脚本，只选择 CONFIG / TASK 以及指定其他的环境变量等

outputs/<dataset>/<experiment>/
├── train/
├── eval/
├── <task>/
└── legacy/
    # 被归档的历史输出
```

### 2.3 多阶段 experiment

```text
configs/<dataset>/<experiment>/
├── stage_1.yaml
├── stage_2.yaml
└── ...
    # 每个 stage 独立 config

scripts/<dataset>/<experiment>/
├── stage_1/
│   └── <task>.sh
├── stage_2/
│   └── <task>.sh
└── ...

outputs/<dataset>/<experiment>/
├── stage_1/
│   └── <task>/
├── stage_2/
│   └── <task>/
└── legacy/
```

Stage 规则：

- 只有在阶段具有独立 config、checkpoint、日志或生命周期时才使用 stage。
- 后续 stage 必须显式指定上游 checkpoint / artifact。
- 禁止通过扫描目录或 mtime 猜测上游输入。

---

## 3. Experiment 命名

推荐：

```text
{model}-{主要配置}-{seed}
```

例如：

```text
qwen3_4b-base-seed42
```

以下变化通常应创建新的 experiment：

- 模型变化；
- 数据协议变化；
- architecture 变化；
- 影响实验语义的核心超参数变化。

以下变化通常不需要创建新的 experiment：

- GPU 数量变化；
- 单机改多机；
- output mode 变化；
- resume 方式变化；
- background / launcher 方式变化。

要求：

- seed 显式写入。
- 不包含空格、`/`、机器名、GPU 数、端口、时间戳等运行时信息。

---

## 4. Config 规范

完整字段以 `example_config.yaml` 为准。

这里只规定各部分职责：

| Config 区域 | 作用 |
|---|---|
| `launch.run` | project、dataset、experiment、stage、workspace/data root |
| `launch.resources` | GPU、nnodes、node rank、master 等默认资源 |
| `launch.outputs` | workspace / data 的 experiment 级输出根目录 |
| `tasks.<task>.launch` | module、distributed、output、resume、background 等 launcher 行为 |
| `tasks.<task>.experiment` | seed、epoch、batch size、learning rate 等实验参数 |

多节点时，launcher 至少允许环境变量覆盖：

```text
NNODES
NODE_RANK
MASTER_ADDR
MASTER_PORT
NPROC_PER_NODE
CUDA_VISIBLE_DEVICES
```

节点专用网卡、环境和路径由 `resource.md` / 环境变量提供。

---

## 5. Task、Shell 与 Launcher 职责

### 5.1 Task module

task module 必须可以通过：

```bash
python -m <module>
```

启动。

task module 负责具体业务计算，并通过 launcher 注入的环境变量获取运行时信息。

task module 应支持以下环境变量：

| 环境变量                    | 含义                             | 规范                                               |
| ----------------------- | ------------------------------ | ------------------------------------------------ |
| `TASK`                  | 当前 task 名，例如 `train`、`eval`    | 由 launcher 注入；task module 不自行猜测                  |
| `CONFIG`                | 当前实际使用的 config 文件路径            | 用于读取当前实验配置                                       |
| `LARGE_DATA_OUTPUT_DIR` | 大文件输出目录                        | checkpoint、大型 artifact、prediction、cache 等必须写入此目录 |
| `LIGHT_DATA_OUTPUT_DIR` | 轻量输出目录                         | log、metrics、summary 等轻量文件写入此目录                   |
| `RANK`                  | global rank                    | 分布式任务由 `torchrun` 提供；非分布式任务按 `0` 处理              |
| `LOCAL_RANK`            | 当前节点内 rank                     | 分布式任务由 `torchrun` 提供；非分布式任务按 `0` 处理              |
| `WORLD_SIZE`            | 全局 worker 数                    | 分布式任务由 `torchrun` 提供；非分布式任务按 `1` 处理              |
| `RESUME`                | 当前是否为 resume 启动                | 使用 launcher 最终解析后的值，不根据 checkpoint 是否存在自行判断      |
| `CHECKPOINT`            | launcher 最终解析得到的 checkpoint 路径 | resume 时直接使用；为空时不得自行扫描目录寻找 checkpoint            |

其中：

- `LARGE_DATA_OUTPUT_DIR` 和 `LIGHT_DATA_OUTPUT_DIR` 是 task module 唯一应依赖的最终输出路径。
- task module 不得根据 `workspace_root`、`data_root`、`output_mode`、`stage` 等字段再次自行推导最终输出目录。
- checkpoint、大型 artifact 等写入 `LARGE_DATA_OUTPUT_DIR`。
- log、metrics、summary 等轻量文件写入 `LIGHT_DATA_OUTPUT_DIR`。
- `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 优先遵循 `torchrun` 的标准环境变量语义。
- 全局共享文件仍遵循 distributed 规范：默认只由 global rank 0 写入。
- `CONFIG` 用于读取实验参数；launcher runtime override 以 launcher 注入的环境变量/参数为准

task module 不负责：

- output path 派生；
- `output_mode` 解析；
- existing output 的 archive / delete；
- checkpoint 自动查找；
- pid / status 管理；
- background / `nohup`；
- SSH 多节点编排；
- master / node rank 分配。

这些由 launcher 统一负责。

### 5.2 Task module 输出

一些输出路径定义：

`<workspace_output_root>` = `<workspace_root>/<project>/outputs/<dataset>/<experiment>/`
`<data_output_root>` = `<data_root>/<project>/<dataset>/<experiment>/`

对于单阶段：
`<workspace_output_dir>` = `<workspace_output_root>/<task>/`
`<data_output_dir>` = `<data_output_root>/<task>/`
对于多阶段：
`<workspace_output_dir>` = `<workspace_output_root>/stage_<N>/<task>/`
`<data_output_dir>` = `<data_output_root>/stage_<N>/<task>/`

以上所有均由 Launcher 根据 config 得到

使用两个环境变量告诉 task module 输出目录：`LARGE_DATA_OUTPUT_DIR` 与 `LIGHT_DATA_OUTPUT_DIR`，均由 Launcher 根据 config 得到并以环境变量的形式注入给 task module

| environment variable    | 含义                           | 规范                                               |
| ----------------------- | ------------------------------ | ------------------------------------------------ |
| `LARGE_DATA_OUTPUT_DIR` | 大文件输出目录                  | checkpoint、大型 artifact、prediction、cache 等必须写入此目录 |
| `LIGHT_DATA_OUTPUT_DIR` | 轻量输出目录                    | log、metrics、summary 等轻量文件写入此目录                   |

在 Launcher 中，不同 output_mode 的指定会给 `LARGE_DATA_OUTPUT_DIR` 与 `LIGHT_DATA_OUTPUT_DIR` 不同的值

| output_mode | `LARGE_DATA_OUTPUT_DIR`  | `LIGHT_DATA_OUTPUT_DIR` |
|---          |---|---|---|
| both        | `<data_output_dir>`      | `<data_output_dir>` |
| data        | `<data_output_dir>`      | `<data_output_dir>` |
| workspace   | `<workspace_output_dir>` | `<workspace_output_dir>` |

output_mode=both 时 workspace 的轻量文件复制/同步由 launcher 负责

### 5.3 Shell

experiment shell 只负责选择 config / task：

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/<dataset>/<experiment>/config.yaml
TASK=<task>

source tools/launcher/launch_from_config.sh "$@"
```

不得在 shell 中重复写 GPU、module、checkpoint、output、master 或实验超参数。

### 5.4 Launcher

launcher 负责：

| 功能 | 说明 |
|---|---|
| config | 解析 config 和 CLI override |
| output | 解析最终输出路径 |
| existing output | archive / delete / raise |
| background | 前台 / 后台启动 |
| resume | checkpoint 解析 |
| process | pid / status / exit code |
| distributed | torchrun 和多节点启动 |
| dry-run | 打印最终解析结果且无副作用 |

`--dry-run` 至少打印：

- project / dataset / experiment / stage / task；
- module；
- nnodes / node rank / nproc；
- master addr / port；
- output mode；
- large / light output dir；
- resume / checkpoint；
- 最终 command。

`--dry-run` 不得创建或修改 output、pid、lock、status、metadata，也不得 archive / delete。

### 5.5 Launcher 处理 output_mode existing_output background resume

以下行为都由 launcher 负责解析和执行，task module 只使用 launcher 最终提供的 `LARGE_DATA_OUTPUT_DIR`、`LIGHT_DATA_OUTPUT_DIR` 和 `CHECKPOINT`。

#### 5.5.1 output_mode

config：

```yaml
tasks:
  <task>:
    launch:
      output_mode: workspace | data | both
```

可通过：

```text
--output-mode workspace|data|both
```

控制 Launcher 如何赋值环境变量 `LARGE_DATA_OUTPUT_DIR` 与 `LIGHT_DATA_OUTPUT_DIR`

#### 5.5.2 existing_output

config：

```yaml
tasks:
  <task>:
    launch:
      existing_output: archive | delete | raise
```

可通过：

```text
--existing-output archive|delete|raise
```

覆盖。

| 模式        | 行为                              |
| --------- | ------------------------------- |
| `archive` | 将已有 task output 归档到对应 `legacy/` |
| `delete`  | 删除当前解析出的 task output 后重新运行      |
| `raise`   | task output 已存在时直接失败            |

`delete` 只能作用于 launcher 已解析出的 task output，必须防止误删上级目录。

#### 5.5.3 background

config：

```yaml
tasks:
  <task>:
    launch:
      background: true | false
```

可通过：

```text
--background true|false
```

控制 Launcher 如何启动任务（前台还是后台）

| 模式 | 行为 |
|---|---|
| `background=false` | 前台运行，launcher 等待并返回真实 exit code |
| `background=true` | 使用 `nohup`，写日志和 pid，并检查进程没有立即退出 |

#### 5.5.4 resume

config：

```yaml
tasks:
  <task>:
    launch:
      resume:
        checkpoint: null
```

是否 resume 由 CLI 控制：

```text
--resume
```

checkpoint 解析优先级：

1. config 中显式指定的 `resume.checkpoint`；
2. `<LARGE_DATA_OUTPUT_DIR>/checkpoints/latest.pt`；
3. `<LARGE_DATA_OUTPUT_DIR>/checkpoints/best.pt`。

launcher 解析完成后通过 `RESUME=0 | 1` 和 `CHECKPOINT` 提供给 task module。

如果指定 `--resume` 但找不到 checkpoint，必须在启动 task module 前失败。

禁止扫描整个 experiment、按 mtime 猜 checkpoint、resume 失败后自动从头运行或隐式跨 stage resume。

## 6. 输出

### 6.1 输出结构

每个 task 的逻辑输出结构统一为：

```text
<output_dir>/
├── configs/
│   # 本次运行实际使用的 resolved config / config snapshot
│
├── logs/
│   ├── <task>.log
│   ├── <task>.exitcode
│   ├── <task>.status.json
│   └── ...
│
├── checkpoints/
│   # latest.pt、best.pt 及其他 checkpoint
│
├── manifest.json
│   # 本次运行的主要产物和运行信息索引
│
├── command.txt
│   # 实际执行的完整 command
│
└── environment.json
    # Python、PyTorch、CUDA、GPU、hostname、git 等环境信息
```

实际文件位置遵循 launcher 提供的输出目录：

- `checkpoints/`、大型 prediction / artifact 等写入 `LARGE_DATA_OUTPUT_DIR`；
- `configs/`、`logs/`、`manifest.json`、`command.txt`、`environment.json` 等轻量文件写入 `LIGHT_DATA_OUTPUT_DIR`；

当 large / light 指向同一目录时，上述结构表现为一棵完整目录树。

### 6.2 状态与多节点记录

launcher 至少维护：

```text
logs/<task>.exitcode
logs/<task>.status.json
```

多节点任务的 status 至少记录每个节点：

```text
host
node_rank
pid
log
exit_status
```

任务最终状态必须综合所有节点结果；任一节点不可恢复失败时，整个 job 不能标记为成功。

### 6.3 日志

正式主日志和全局 `metrics.jsonl` 默认只由 global rank 0 写。

推荐使用稳定的单行格式：

```text
2026-08-22T14:30:12+08:00 | INFO | rank=0 | phase=train | epoch=3/25 | iter=120/500 | global_step=1120 | loss=0.1832 | lr=1.82e-4 | data_time_ms=12.4 | step_time_ms=238.7 | samples_per_s=134.1
```

训练日志至少记录：

- epoch、iter、global step；
- loss 及主要 loss 项；
- learning rate、grad norm（如适用）；
- data time、step time、throughput；
- checkpoint 保存事件。

eval 日志至少记录：

- split；
- 样本数；
- 使用的 checkpoint；
- duration；
- 主要指标。

分布式指标必须先聚合，再由 rank 0 写入主日志和 `metrics.jsonl`。

NaN、OOM、通信异常和非零退出必须记录 rank、phase、step 等上下文，不能只保留 traceback。
