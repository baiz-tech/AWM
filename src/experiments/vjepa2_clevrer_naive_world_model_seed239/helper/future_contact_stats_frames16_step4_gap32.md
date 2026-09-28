# Future 范围内接触标签统计

标签定义：若以下连续原始帧区间内任意一帧的 `labels.target_contacting_zone=True`，则样本为正例：

```text
[P+32, P+96)
共 64 个连续原始帧
```

其中 `P=static.start_frame_for_prediction`。区间采用左闭右开；统计的是整个连续 Future 时间范围，而不只是按 step 采样的 16 帧。

## 配置

```yaml
root: /data/ABDUCTIVE-WORLD/physion_v2/extracted
splits: ['data_v1', 'readout_data_v1', 'testdata_v1']
clip_frames: 16
future_frame_step: 4
clip_gap: 32
raw_future_range_length: 64
```

## 汇总

| Split | 发现视频 | 有效总样本 | 正例 | 负例 | 正例率 | 无效样本 |
|---|---:|---:|---:|---:|---:|---:|
| `data_v1` | 8000 | 7986 | 2274 | 5712 | 28.47% | 14 |
| `readout_data_v1` | 800 | 797 | 202 | 595 | 25.35% | 3 |
| `testdata_v1` | 1190 | 1189 | 354 | 835 | 29.77% | 1 |

`total_samples = positive_samples + negative_samples`。缺少 PKL、metadata 异常或 Future 连续区间不完整的样本计入无效样本，不会被误算为负例。

## 输出文件

- JSON（含错误类型、无效样本示例和首次接触 offset 分布）：`future_contact_stats_frames16_step4_gap32.json`
- CSV 汇总：`future_contact_stats_frames16_step4_gap32.csv`
