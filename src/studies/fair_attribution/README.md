# EK100 公平归因矩阵（E0–E4）

本实验只训练统一的 `AttributionHead`，所有特征抽取器冻结，使用相同的
EK100 cache、标签、seed 和训练预算。E0–E4 的区别只在特征路径：

```text
E0 raw grid → fixed pooling → head
E1 raw grid → readout.world → head
E2 raw grid → shared.world → head
E3 raw grid → shared residual → frozen readout.world → head
E4 E3 feature + shared Entity/Dynamic/Relation state → head
```

E0 的 `1280 → 512` 是无参数的分组平均；head 输入、结构和参数量在五个版本中
完全相同。E3 用来估计第二次 world extraction 的贡献，E4 用来估计完整 residual
HASP 接口的额外贡献。该实验不修改 `reproduce_v23_ek100_only`。

训练前需要已有 V23 `readout/best.pt` 和 `adapter/best.pt`，以及 train/validation
cache。先运行参数检查和 launcher 的 `--dry-run`。

## 产物布局

五个 variant 写入**同一个 artifact 根**，`evaluate` 才能在单个
`--checkpoints` 目录下找到全部五个 head：

```text
{experiment_root}/E0/best.pt ... {experiment_root}/E4/best.pt
{experiment_root}/evaluate/metrics.json
{experiment_root}/parameter_report/parameters.json
```

`train_e0 ... train_e4` 的 `--output` 因此是 `{experiment_root}/E0 ...`，而不是各自
的 `{large_output_dir}`——后者只保存 log、manifest 等运行元数据。

`evaluate.py` 接受两种布局：`<checkpoints>/E0.pt`（迁移前的扁平布局）或
`<checkpoints>/E0/best.pt`（当前 launcher 布局）。两种都不存在时会明确报出
两个被查找过的路径，不会静默跳过某个 variant。
