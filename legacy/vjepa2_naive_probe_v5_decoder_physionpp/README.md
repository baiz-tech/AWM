# Physion++ decoder utilities

`scripts/visualize_physion_clip.py` uses `seed` to deterministically select one
video from the sorted videos in the requested Physion++ split, then renders the
native current+future clip selected from that video's paired PKL
`static.start_frame_for_prediction`.
It writes three videos:

- `original.mp4`: selected RGB frames;
- `projected_3d_boxes.mp4`: projection of the PKL 3D object boxes using the
  per-frame camera and projection matrices;
- `segmentation_boxes.mp4`: 2D boxes from the paired `_id.json` RLE
  segmentation annotations. `--segmentation-source color` enables explicit
  RGB segmentation-color matching instead.

All MP4 outputs are transcoded to browser-compatible H.264 (`yuv420p`) with
fast-start metadata. The manifest records `"video_codec": "h264"`.

Example:

```bash
conda activate vjepa2-312
python recipe/vjepa2_naive_probe_v5_decoder_physionpp/scripts/visualize_physion_clip.py \
  --dataset-root /data/ABDUCTIVE-WORLD/physion_v2/extracted \
  --split data_v1 \
  --seed 239
```

未指定 `--output-dir` 时，输出写入
`outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp/<video_name>_seed239/`，并
保存 `manifest.json` 记录 seed、视频、帧索引和输出文件。当前可视化本身是
确定性的；同一个 `dataset-root`、`split`、`video-glob` 和 seed 会选择同一个
视频，并在 manifest 中记录候选总数和选择索引。

脚本不会覆盖已有结果。如果同一路径中的三个视频和 manifest 均完整、且输入
协议一致，重复执行会直接返回已有 manifest，并标记
`"existing_complete": true`；如果目录不完整或 manifest 冲突，则要求更换
seed 或显式指定新的 `--output-dir`。

The default protocol is 16 current frames at raw step 2 immediately before
`P`, followed by 16 future frames at raw step 4 from `P`, for 32 output frames.
Use `--segmentation-source id-json|color|auto` to control the third output.
