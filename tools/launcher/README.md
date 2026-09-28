# tools/launcher/

公共启动逻辑。实验脚本只选择 `CONFIG` 与 `TASK`,其余全部由 launcher 负责。

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/<dataset>/<experiment>/config.yaml
TASK=<task>

source tools/launcher/launch_from_config.sh "$@"
```

从仓库根目录执行。

## 文件

| 文件 | 作用 |
|---|---|
| `launch_from_config.sh` | launcher 主体:解析 config/CLI、解析输出路径、处理 existing output、resume、后台运行、distributed 启动、写 manifest |
| `render_task.py` | 为单个 task 生成 task-resolved config,并插值 `args` / `env` |

## CLI

```text
--output-mode workspace|data|both
--existing-output archive|delete|raise
--background true|false
--resume
--dry-run
-- <其余参数原样传给 task module>
```

`--dry-run` 打印解析结果与最终 command,**不创建或修改任何文件**。

## 输出路径

```text
<workspace_output_root> = {workspace_root}/{dataset}/{experiment}/
<data_output_root>      = {data_root}/{project}/{dataset}/{experiment}/
单阶段:  <root>/<task>/
多阶段:  <root>/stage_<N>/<task>/
```

`{experiment_root}` 为 task 输出目录的上一级;后续阶段通过
`{experiment_root}/<上游 task>/` 显式引用上游产物。

| output_mode | `LARGE_DATA_OUTPUT_DIR` | `LIGHT_DATA_OUTPUT_DIR` |
|---|---|---|
| `both` | data task dir | data task dir(镜像到 workspace) |
| `data` | data task dir | data task dir |
| `workspace` | workspace task dir | workspace task dir |

## 注入给 task module 的环境变量

```text
TASK CONFIG
LARGE_DATA_OUTPUT_DIR LIGHT_DATA_OUTPUT_DIR
RESUME CHECKPOINT
RANK LOCAL_RANK WORLD_SIZE        (分布式由 torchrun 提供)
```

以及 `LAUNCH_*` 前缀的运行上下文(`LAUNCH_EXPERIMENT_ROOT`、
`LAUNCH_RESOLVED_CONFIG` 等),供 task module 需要时读取。

## 参数由模块侧解析

`tasks.<task>.launch.args` 现在统一为空:参数改由模块通过
`src/core/run_context.py` 从 `tasks.<task>.experiment.cli_defaults` 取默认值,
这正是 `requirement.md` §5.1 要求的形态。launcher 仍然支持 `args`,并会做同样的
模板渲染,用于临时追加或覆盖参数。

`cli_defaults` 的写法与 `args` 使用同一套模板变量:

```yaml
tasks:
  train_probe:
    launch:
      module: src.experiments.<exp>.train_probe
      args: []
    experiment:
      seed: 239
      cli_defaults:
        --train-cache: "{experiment_root}/cache_train"
        --output-dir: "{large_output_dir}"
        --epochs: "30"
        --seed: "{seed}"
```

可用变量:

```text
{config} {resolved_config} {task} {project} {dataset} {experiment} {stage}
{experiment_root} {data_experiment_root} {workspace_experiment_root}
{large_output_dir} {light_output_dir} {checkpoint} {seed}
{nnodes} {nproc} {master_addr} {master_port}
{cfg:<dotted.path>}        从原始 config 取值
```

渲染失败(变量未知、`{cfg:...}` 解析不到)会让 launcher 直接报错退出,不会
静默传空字符串。

## 任务级 resolved config

```yaml
tasks:
  train_probe:
    launch:
      # 可选:以某个文件为基准(用于结构特殊的模块,如 flat predictor config)
      config_template: src/experiments/<exp>/configs_legacy/predictor_config.yaml
      # 可选:在基准之上覆盖字段,值同样支持插值
      config_overrides:
        folder: "{large_output_dir}"
        meta.seed: 239
      # 可选:模块从 tasks.<alias>.experiment 读取参数时指定别名,默认 train
      config_alias_task: train
      # 可选:渲染后会被 source 的环境变量
      env:
        PROBE_CHECKPOINT: "{experiment_root}/train_probe/best.pt"
```

launcher 会写出:

```text
<light_output_dir>/configs/<task>.resolved.yaml
<light_output_dir>/configs/<config 文件名>          # 原始 config 快照
<light_output_dir>/configs/<task>.env               # 仅在声明了 env 时
```

resolved config 同时提供三种读取方式,以兼容历史上不同时期写的模块:

```yaml
seed: 239                 # 1. 顶层扁平字段(config["data"] / config["meta"] ...)
folder: <task 输出目录>
launch: {...}
tasks:
  train:                  # 2. 直接字典访问 config["tasks"]["train"]["experiment"]
    experiment: {...}
  <task>:                 # 3. config_utils.task_experiment(config, "<task>")
    experiment: {...}
```

## 状态与复现信息

```text
command.txt             实际执行的完整命令
manifest.json           运行索引(含 resolved_config 与解析后的输出目录)
environment.json        Python / torch / CUDA / GPU / hostname / git 状态
logs/<task>.log         主日志(后台模式)
logs/<task>.exitcode    退出码
logs/<task>.status.json 运行状态
```

## 多节点

launcher 允许用环境变量覆盖 config 默认值:

```text
NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE CUDA_VISIBLE_DEVICES
```

分布式 task 启动前会校验可见 GPU 数不少于 `NPROC_PER_NODE`。
