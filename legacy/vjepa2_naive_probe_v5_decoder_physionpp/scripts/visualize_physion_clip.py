#!/usr/bin/env python3
"""Render a Physion++ clip and GT 2D box overlays.

Outputs original.mp4, projected_3d_boxes.mp4, and segmentation_boxes.mp4.
The default clip follows the native Physion protocol: 16 current frames at
step 2 before P and 16 future frames at step 4 from P, with P read from PKL.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import subprocess
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np


def metadata_for(video: Path) -> dict[str, Any]:
    if not video.name.endswith("_img.mp4"):
        raise ValueError(f"video must end with _img.mp4: {video}")
    path = video.with_name(video.name[:-8] + ".pkl")
    if not path.is_file():
        raise FileNotFoundError(f"missing metadata: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("frames"), dict):
        raise ValueError(f"invalid metadata: {path}")
    return data


def select_video(dataset_root: Path, split: str, seed: int, video_glob: str) -> tuple[Path, int, int]:
    split_root = dataset_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Physion++ split does not exist: {split_root}")
    videos = sorted(path for path in split_root.glob(video_glob) if path.is_file())
    if not videos:
        raise FileNotFoundError(f"no videos found under {split_root} with glob {video_glob!r}")
    selection_index = random.Random(seed).randrange(len(videos))
    return videos[selection_index], selection_index, len(videos)


def load_id_annotations(video: Path) -> dict[str, Any] | None:
    path = video.with_name(video.name[:-8] + "_id.json")
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid segmentation annotation: {path}")
    return data


def frame_meta(data: dict[str, Any], index: int) -> dict[str, Any]:
    frames = data["frames"]
    return frames.get(f"{index:04d}", frames.get(index))


def clip_indices(data: dict[str, Any], count: int, current_step: int,
                 future_step: int, gap: int) -> tuple[int, np.ndarray]:
    try:
        anchor = int(data["static"]["start_frame_for_prediction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("PKL lacks static.start_frame_for_prediction") from exc
    current = anchor - count * current_step + np.arange(count) * current_step
    future = anchor + gap + np.arange(count) * future_step
    indices = np.concatenate((current, future)).astype(np.int64)
    available = np.asarray([int(k) for k in data["frames"]], dtype=np.int64)
    if indices.min() < available.min() or indices.max() > available.max():
        raise ValueError(f"requested frames {indices.min()}..{indices.max()} outside metadata range")
    return anchor, indices


def names_and_colors(data: dict[str, Any], count: int) -> tuple[list[str], np.ndarray]:
    static = data.get("static", {})
    names = []
    for value in list(static.get("model_names", []))[:count]:
        names.append(value.decode(errors="replace") if isinstance(value, bytes) else str(value))
    names += [f"object_{i}" for i in range(len(names), count)]
    values = static.get("video_object_segmentation_colors", static.get("object_segmentation_colors"))
    colors = np.asarray(values, dtype=np.int16)
    if colors.ndim != 2 or colors.shape[1:] != (3,) or len(colors) < count:
        raise ValueError("PKL lacks enough RGB segmentation colors")
    return names, colors[:count]


def count_objects(annotation: dict[str, Any]) -> int:
    positions = np.asarray(annotation["objects"]["positions"])
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("objects.positions must have shape [N,3]")
    return len(positions)


def corners(annotation: dict[str, Any], index: int) -> np.ndarray:
    obj = annotation["objects"]
    try:
        left, right = np.asarray(obj["left"])[index], np.asarray(obj["right"])[index]
        bottom, top = np.asarray(obj["bottom"])[index], np.asarray(obj["top"])[index]
        back, front = np.asarray(obj["back"])[index], np.asarray(obj["front"])[index]
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("frame lacks 3D object corners") from exc
    return np.asarray([[x, y, z] for x in (left[0], right[0])
                       for y in (bottom[1], top[1])
                       for z in (back[2], front[2])], dtype=np.float32)


def projected_box(annotation: dict[str, Any], index: int, width: int, height: int):
    matrices = annotation["camera_matrices"]
    view = np.asarray(matrices["camera_matrix"], dtype=np.float32).reshape(4, 4)
    projection = np.asarray(matrices["projection_matrix"], dtype=np.float32).reshape(4, 4)
    points = np.c_[corners(annotation, index), np.ones(8, dtype=np.float32)]
    clip = (projection @ view @ points.T).T
    valid = np.abs(clip[:, 3]) > 1e-6
    if not valid.any():
        return None
    ndc = clip[valid, :3] / clip[valid, 3:4]
    pixels = np.c_[(ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2]
    lo, hi = np.floor(pixels.min(0)).astype(int), np.ceil(pixels.max(0)).astype(int)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def segmentation_box(frame: np.ndarray, rgb: np.ndarray, tolerance: int):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.int16)
    mask = np.abs(image - rgb.reshape(1, 1, 3)).max(2) <= tolerance
    ys, xs = np.where(mask)
    return None if not len(xs) else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    height, width = (int(value) for value in rle["size"])
    encoded = rle["counts"]
    counts, position = [], 0
    while position < len(encoded):
        value, shift = 0, 0
        while True:
            code = ord(encoded[position]) - 48
            position += 1
            value |= (code & 31) << (5 * shift)
            if not code & 32:
                if code & 16:
                    value |= -1 << (5 * (shift + 1))
                break
            shift += 1
        counts.append(value + counts[-2] if len(counts) > 2 else value)
    flat, offset = np.zeros(height * width, dtype=np.uint8), 0
    for index, run in enumerate(counts):
        if run < 0 or offset + run > flat.size:
            raise ValueError("invalid segmentation RLE")
        if index % 2:
            flat[offset:offset + run] = 1
        offset += run
    if offset != flat.size:
        raise ValueError("segmentation RLE does not cover image")
    return flat.reshape((height, width), order="F").astype(bool)


def id_json_box(id_annotations: dict[str, Any], frame_index: int, object_index: int):
    entries = id_annotations.get(f"{frame_index:04d}", id_annotations.get(frame_index, []))
    for entry in entries:
        if int(entry.get("idx", -1)) != object_index:
            continue
        mask = decode_rle(entry)
        ys, xs = np.where(mask)
        return None if not len(xs) else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return None


def draw(frame: np.ndarray, box, color: tuple[int, int, int], label: str) -> None:
    if box is None:
        return
    x0, y0, x1, y1 = box
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1, cv2.LINE_AA)


def writer(path: Path, fps: float, size: tuple[int, int]):
    result = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not result.isOpened():
        raise RuntimeError(f"failed to open writer: {path}")
    return result


def encode_h264(source: Path, target: Path) -> None:
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
    ], check=True)
    source.unlink()


def render(args: argparse.Namespace) -> dict[str, Any]:
    video = args.video.resolve()
    data = metadata_for(video)
    id_annotations = load_id_annotations(video)
    anchor, indices = clip_indices(data, args.clip_frames, args.current_step, args.future_step, args.clip_gap)
    first = frame_meta(data, int(indices[0]))
    count = count_objects(first)
    names, colors = names_and_colors(data, count)
    paths = [args.output_dir / x for x in ("original.mp4", "projected_3d_boxes.mp4", "segmentation_boxes.mp4")]
    manifest_path = args.output_dir / "manifest.json"
    if all(path.is_file() for path in [*paths, manifest_path]):
        with manifest_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        expected = {
            "seed": args.seed,
            "split": args.split,
            "video": str(video),
            "frame_indices": indices.tolist(),
            "segmentation_source": args.segmentation_source,
            "video_codec": "h264",
        }
        if all(existing.get(key) == value for key, value in expected.items()):
            existing["existing_complete"] = True
            return existing
        raise FileExistsError(
            f"output run exists with a different manifest; choose another seed or --output-dir: {manifest_path}"
        )
    occupied = [path for path in [*paths, manifest_path] if path.exists()]
    if occupied:
        raise FileExistsError(f"output run is incomplete; choose another seed or --output-dir: {occupied[0]}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = [args.output_dir / f".{path.stem}.mp4v.mp4" for path in paths]
    outputs = [writer(path, fps, (width, height)) for path in temporary_paths]
    try:
        for position, raw_index in enumerate(indices.tolist()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, raw_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed reading frame {raw_index}")
            annotation = frame_meta(data, raw_index)
            projected, segmented = frame.copy(), frame.copy()
            for i, name in enumerate(names):
                label = f"{i}:{name}"
                draw(projected, projected_box(annotation, i, width, height), (255, 0, 255), label)
                if args.segmentation_source == "id-json":
                    if id_annotations is None:
                        raise FileNotFoundError("--segmentation-source id-json requires the paired _id.json")
                    box = id_json_box(id_annotations, raw_index, i)
                elif args.segmentation_source == "color":
                    box = segmentation_box(frame, colors[i], args.color_tolerance)
                else:
                    box = (id_json_box(id_annotations, raw_index, i)
                           if id_annotations is not None else segmentation_box(frame, colors[i], args.color_tolerance))
                draw(segmented, box, (0, 220, 0), label)
            phase = "current" if position < args.clip_frames else "future"
            text = f"frame={raw_index} {phase} P={anchor}"
            for image in (projected, segmented):
                cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
            outputs[0].write(frame); outputs[1].write(projected); outputs[2].write(segmented)
    finally:
        cap.release()
        for output in outputs:
            output.release()
    for source, target in zip(temporary_paths, paths):
        encode_h264(source, target)
    result = {"seed": args.seed, "dataset_root": str(args.dataset_root.resolve()),
              "split": args.split, "video_glob": args.video_glob,
              "video_selection_index": args.video_selection_index,
              "video_candidate_count": args.video_candidate_count,
              "video": str(video), "anchor": anchor,
              "frame_indices": indices.tolist(), "clip_frames": args.clip_frames,
              "current_step": args.current_step, "future_step": args.future_step,
              "clip_gap": args.clip_gap, "segmentation_source": args.segmentation_source,
              "color_tolerance": args.color_tolerance, "video_codec": "h264",
              "outputs": [str(path) for path in paths]}
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/data/ABDUCTIVE-WORLD/physion_v2/extracted"))
    parser.add_argument("--split", choices=("data_v1", "readout_data_v1", "testdata_v1"), default="data_v1")
    parser.add_argument("--video-glob", default="**/*_img.mp4")
    parser.add_argument("--output-dir", type=Path,
                        help="default: outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp/<video>_seed<seed>")
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--current-step", type=int, default=2)
    parser.add_argument("--future-step", type=int, default=4)
    parser.add_argument("--clip-gap", type=int, default=0)
    parser.add_argument("--color-tolerance", type=int, default=8)
    parser.add_argument("--segmentation-source", choices=("auto", "id-json", "color"), default="auto",
                        help="use paired *_id.json RLE masks (default when available) or RGB color matching")
    args = parser.parse_args()
    if min(args.clip_frames, args.current_step, args.future_step) <= 0 or args.clip_gap < 0:
        parser.error("clip-frames/current-step/future-step must be positive; clip-gap non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.video, args.video_selection_index, args.video_candidate_count = select_video(
        args.dataset_root, args.split, args.seed, args.video_glob
    )
    if args.output_dir is None:
        repo_root = Path(__file__).resolve().parents[3]
        video_name = args.video.name[:-8] if args.video.name.endswith("_img.mp4") else args.video.stem
        args.output_dir = repo_root / "outputs" / "runs" / "vjepa2_naive_probe_v5_decoder_physionpp" / f"{video_name}_seed{args.seed}"
    print(json.dumps(render(args), indent=2))


if __name__ == "__main__":
    main()
