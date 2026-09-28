#!/usr/bin/env python3
"""Count horizon-local OCP labels in Physion++ splits.

The contact label is one when ``target_contacting_zone`` is true in any raw
frame in the fixed half-open Future interval

    [P + clip_gap, P + clip_gap + clip_frames * future_frame_step)

where P is ``static.start_frame_for_prediction``. With
``--all-future-frames``, the interval instead ends at the end of each video.
Physion++ PKLs are trusted local dataset files; never run this program on
untrusted pickle files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path


CONTACT_KEY = "target_contacting_zone"
VIDEO_SUFFIX = "_img.mp4"


@dataclass(frozen=True)
class SampleResult:
    status: str
    positive: bool = False
    first_contact_frame: int | None = None
    first_contact_offset: int | None = None
    detail: str = ""


def pkl_path_for_video(video_path: str | Path) -> Path:
    path = Path(video_path)
    if not path.name.endswith(VIDEO_SUFFIX):
        raise ValueError(f"video name must end with {VIDEO_SUFFIX!r}: {path}")
    return path.with_name(path.name[: -len(VIDEO_SUFFIX)] + ".pkl")


def _frame_at(frames, index: int):
    """Support integer and common zero-padded/string frame keys."""
    frame = frames.get(index)
    if frame is None:
        frame = frames.get(str(index))
    if frame is None:
        frame = frames.get(f"{index:04d}")
    return frame


def label_video(video_path: str, clip_frames: int, future_frame_step: int,
                clip_gap: int, all_future_frames: bool = False) -> SampleResult:
    pkl_path = pkl_path_for_video(video_path)
    if not pkl_path.is_file():
        return SampleResult("missing_pkl", detail=str(pkl_path))
    try:
        with pkl_path.open("rb") as handle:
            metadata = pickle.load(handle)
        prediction_start = int(metadata["static"]["start_frame_for_prediction"])
        frames = metadata["frames"]
        if not frames:
            return SampleResult("invalid_metadata", detail="empty frames")
    except (OSError, EOFError, KeyError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        return SampleResult("invalid_metadata", detail=repr(exc))

    range_start = prediction_start + clip_gap
    if all_future_frames:
        try:
            range_end = max(int(index) for index in frames) + 1
        except (TypeError, ValueError) as exc:
            return SampleResult("invalid_metadata", detail=f"invalid frame keys: {exc!r}")
        if range_start >= range_end:
            return SampleResult(
                "incomplete_range",
                detail=f"Future start {range_start} is not before video end {range_end}",
            )
    else:
        range_end = range_start + clip_frames * future_frame_step
    missing = [index for index in range(range_start, range_end) if _frame_at(frames, index) is None]
    if missing:
        return SampleResult(
            "incomplete_range",
            detail=(
                f"required=[{range_start},{range_end}) missing_count={len(missing)} "
                f"first_missing={missing[0]}"
            ),
        )

    first_contact = None
    for index in range(range_start, range_end):
        labels = _frame_at(frames, index).get("labels", {})
        if bool(labels.get(CONTACT_KEY, False)):
            first_contact = index
            break
    return SampleResult(
        "valid",
        positive=first_contact is not None,
        first_contact_frame=first_contact,
        first_contact_offset=None if first_contact is None else first_contact - range_start,
    )


def _worker(arguments):
    path, clip_frames, future_frame_step, clip_gap, all_future_frames = arguments
    try:
        return path, label_video(
            path, clip_frames, future_frame_step, clip_gap, all_future_frames
        )
    except Exception as exc:  # Keep one corrupt sample from terminating a full scan.
        return path, SampleResult("unexpected_error", detail=repr(exc))


def _percentiles(values):
    if not values:
        return {}
    ordered = sorted(values)

    def percentile(q):
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        fraction = position - lower
        return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)

    return {
        "min": float(ordered[0]),
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


def scan_split(root: Path, split: str, video_glob: str, clip_frames: int,
               future_frame_step: int, clip_gap: int, all_future_frames: bool,
               workers: int):
    paths = sorted((root / split).glob(video_glob))
    arguments = [
        (str(path), clip_frames, future_frame_step, clip_gap, all_future_frames)
        for path in paths
    ]
    if workers == 1:
        records = map(_worker, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        records = executor.map(_worker, arguments, chunksize=4)

    status_counts = {}
    positives = 0
    contact_offsets = []
    invalid_examples = []
    try:
        for path, result in records:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
            if result.status == "valid":
                positives += int(result.positive)
                if result.first_contact_offset is not None:
                    contact_offsets.append(result.first_contact_offset)
            elif len(invalid_examples) < 20:
                invalid_examples.append({"path": path, **asdict(result)})
    finally:
        if workers != 1:
            executor.shutdown()

    valid = status_counts.get("valid", 0)
    negatives = valid - positives
    return {
        "split": split,
        "discovered_videos": len(paths),
        "total_samples": valid,
        "positive_samples": positives,
        "negative_samples": negatives,
        "positive_rate": None if valid == 0 else positives / valid,
        "negative_rate": None if valid == 0 else negatives / valid,
        "invalid_samples": len(paths) - valid,
        "status_counts": status_counts,
        "positive_first_contact_offset_stats": _percentiles(contact_offsets),
        "invalid_examples": invalid_examples,
    }


def write_csv(path: Path, summaries):
    fields = (
        "split", "discovered_videos", "total_samples", "positive_samples",
        "negative_samples", "positive_rate", "negative_rate", "invalid_samples",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary[key] for key in fields})


def write_markdown(path: Path, report, json_path: Path, csv_path: Path):
    config = report["configuration"]
    start = config["clip_gap"]
    if config["all_future_frames"]:
        range_text = f"[P+{start}, video_end)"
        length_text = "长度随视频而变化"
    else:
        end = start + config["clip_frames"] * config["future_frame_step"]
        range_text = f"[P+{start}, P+{end})"
        length_text = f"共 {end - start} 个连续原始帧"
    lines = [
        "# Future 范围内接触标签统计",
        "",
        "标签定义：若以下连续原始帧区间内任意一帧的 "
        "`labels.target_contacting_zone=True`，则样本为正例：",
        "",
        "```text",
        range_text,
        length_text,
        "```",
        "",
        "其中 `P=static.start_frame_for_prediction`。区间采用左闭右开；"
        "统计的是整个连续 Future 时间范围，而不只是按 step 采样的 16 帧。",
        "",
        "## 配置",
        "",
        "```yaml",
        f"root: {config['root']}",
        f"splits: {config['splits']}",
        f"clip_frames: {config['clip_frames']}",
        f"future_frame_step: {config['future_frame_step']}",
        f"clip_gap: {config['clip_gap']}",
        f"all_future_frames: {str(config['all_future_frames']).lower()}",
        f"raw_future_range_length: {config['raw_future_range_length']}",
        "```",
        "",
        "## 汇总",
        "",
        "| Split | 发现视频 | 有效总样本 | 正例 | 负例 | 正例率 | 无效样本 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["splits"]:
        rate = "N/A" if item["positive_rate"] is None else f"{100.0 * item['positive_rate']:.2f}%"
        lines.append(
            f"| `{item['split']}` | {item['discovered_videos']} | {item['total_samples']} | "
            f"{item['positive_samples']} | {item['negative_samples']} | {rate} | "
            f"{item['invalid_samples']} |"
        )
    lines.extend([
        "",
        "`total_samples = positive_samples + negative_samples`。缺少 PKL、metadata "
        "异常或 Future 连续区间不完整的样本计入无效样本，不会被误算为负例。",
        "",
        "## 输出文件",
        "",
        f"- JSON（含错误类型、无效样本示例和首次接触 offset 分布）：`{json_path.name}`",
        f"- CSV 汇总：`{csv_path.name}`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("/data/ABDUCTIVE-WORLD/physion_v2/extracted"),
    )
    parser.add_argument(
        "--splits", nargs="+",
        default=("data_v1", "readout_data_v1", "testdata_v1"),
    )
    parser.add_argument("--video-glob", default="**/*_img.mp4")
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--future-frame-step", type=int, default=2)
    parser.add_argument("--clip-gap", type=int, default=0)
    parser.add_argument(
        "--all-future-frames",
        action="store_true",
        help=(
            "Ignore the configured fixed Future length and check every raw frame "
            "from P+clip_gap through the end of the video."
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-prefix", default=None,
        help="Output basename without extension; defaults to a parameterized name.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    if args.clip_frames <= 0 or args.future_frame_step <= 0:
        parser.error("clip-frames and future-frame-step must be positive")
    if args.clip_gap < 0 or args.workers <= 0:
        parser.error("clip-gap must be non-negative and workers must be positive")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_prefix:
        prefix = args.output_prefix
    elif args.all_future_frames:
        prefix = f"future_contact_stats_allfuture_gap{args.clip_gap}"
    else:
        prefix = (
            f"future_contact_stats_frames{args.clip_frames}_step{args.future_frame_step}"
            f"_gap{args.clip_gap}"
        )
    summaries = []
    for split in args.splits:
        print(f"Scanning {split} ...", flush=True)
        summary = scan_split(
            args.root, split, args.video_glob, args.clip_frames,
            args.future_frame_step, args.clip_gap, args.all_future_frames,
            args.workers,
        )
        summaries.append(summary)
        print(
            f"{split}: total={summary['total_samples']} "
            f"positive={summary['positive_samples']} negative={summary['negative_samples']} "
            f"invalid={summary['invalid_samples']}",
            flush=True,
        )

    report = {
        "label": CONTACT_KEY,
        "range_semantics": "half_open_contiguous_raw_frames",
        "configuration": {
            "root": str(args.root.resolve()),
            "splits": list(args.splits),
            "video_glob": args.video_glob,
            "clip_frames": args.clip_frames,
            "future_frame_step": args.future_frame_step,
            "clip_gap": args.clip_gap,
            "all_future_frames": args.all_future_frames,
            "raw_future_range_length": (
                None if args.all_future_frames
                else args.clip_frames * args.future_frame_step
            ),
            "range_start_formula": "P + clip_gap",
            "range_end_exclusive_formula": (
                "max(metadata.frames) + 1" if args.all_future_frames
                else "P + clip_gap + clip_frames * future_frame_step"
            ),
        },
        "splits": summaries,
    }
    json_path = args.output_dir / f"{prefix}.json"
    csv_path = args.output_dir / f"{prefix}.csv"
    markdown_path = args.output_dir / f"{prefix}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_path, summaries)
    write_markdown(markdown_path, report, json_path, csv_path)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
