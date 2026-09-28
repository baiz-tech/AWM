# Future 范围接触标签统计工具

程序：[count_future_contact_labels.py](count_future_contact_labels.py)

它分别扫描：

```text
data_v1
readout_data_v1
testdata_v1
```

并统计 Future 连续原始帧区间内是否曾出现：

```python
frame["labels"]["target_contacting_zone"] == True
```

标签区间采用左闭右开：

```text
start = P + clip_gap
end   = start + clip_frames × future_frame_step
range = [start, end)
```

其中：

```text
P = metadata["static"]["start_frame_for_prediction"]
```

注意：程序检查的是这个范围内的**每个连续原始帧**，不是只检查按 `future_frame_step` 采样的 16 帧。

## 默认统计

```bash
python recipe/vjepa2_naive/helper/count_future_contact_labels.py
```

默认参数：

```text
clip_frames = 16
future_frame_step = 2
clip_gap = 0
Future range = [P,P+32) = P...P+31
```

当前全量结果：

| Split | 总样本 | 正例 | 负例 | 正例率 |
|---|---:|---:|---:|---:|
| `data_v1` | 8,000 | 1,466 | 6,534 | 18.32% |
| `readout_data_v1` | 800 | 127 | 673 | 15.88% |
| `testdata_v1` | 1,190 | 182 | 1,008 | 15.29% |

完整输出：

- [future_contact_stats_frames16_step2_gap0.md](future_contact_stats_frames16_step2_gap0.md)
- [future_contact_stats_frames16_step2_gap0.csv](future_contact_stats_frames16_step2_gap0.csv)
- [future_contact_stats_frames16_step2_gap0.json](future_contact_stats_frames16_step2_gap0.json)

## 指定 step 和 gap

例如原 gap32 配置：

```bash
python recipe/vjepa2_naive/helper/count_future_contact_labels.py \
  --future-frame-step 4 \
  --clip-gap 32
```

对应连续标签范围：

```text
[P+32, P+96)
```

输出文件名自动带参数：

```text
future_contact_stats_frames16_step4_gap32.json
future_contact_stats_frames16_step4_gap32.csv
future_contact_stats_frames16_step4_gap32.md
```

## 检查从 Future 起点到视频结束的所有帧

```bash
python recipe/vjepa2_naive/helper/count_future_contact_labels.py \
  --all-future-frames \
  --clip-gap 0
```

标签范围变为：

```text
[P, video_end)
```

若仍希望跳过 `P` 后的前 32 个原始帧：

```bash
python recipe/vjepa2_naive/helper/count_future_contact_labels.py \
  --all-future-frames \
  --clip-gap 32
```

对应：

```text
[P+32, video_end)
```

启用 `--all-future-frames` 后，`clip_frames` 和 `future_frame_step` 不再决定标签区间终点；`clip_gap` 仍决定起点。默认输出文件名为：

```text
future_contact_stats_allfuture_gap0.json
future_contact_stats_allfuture_gap0.csv
future_contact_stats_allfuture_gap0.md
```

其他参数：

```bash
python recipe/vjepa2_naive/helper/count_future_contact_labels.py --help
```

## 有效样本口径

```text
total_samples = positive_samples + negative_samples
```

以下情况不会被当成负例，而会单独计入 `invalid_samples`：

- 缺少配套 PKL；
- PKL metadata 无法读取；
- `start_frame_for_prediction` 缺失；
- 所需连续 Future 区间中的 frame record 不完整。

JSON 会保存 status 计数、最多 20 个无效样本示例，以及正例首次接触相对 Future 起点的 offset 分布。
