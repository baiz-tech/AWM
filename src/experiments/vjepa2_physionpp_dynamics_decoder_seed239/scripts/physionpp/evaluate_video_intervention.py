#!/usr/bin/env python3
"""Single-clip video interventions for the Physion++ shallow probe.

The script edits the decoded RGB frames, reruns the frozen encoder/predictor
and shallow probe, and writes raw outputs plus paired summaries.  It is
intentionally single-clip first: use it to inspect the protocol before
scaling to a validation set.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from decord import VideoReader, cpu

from external.vjepa2.app.vjepa.transforms import make_transforms
from src.experiments.vjepa2_clevrer_naive_world_model_seed239.model import build_model
from .model import PhysionDecoder, match_objects
from .prepare_targets import decode_rle, targets


def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


def _read_frames(path: Path, indices: np.ndarray) -> np.ndarray:
    reader = VideoReader(str(path), num_threads=1, ctx=cpu(0))
    if int(indices.min()) < 0 or int(indices.max()) >= len(reader):
        raise ValueError(f"frame indices outside {path}: {indices.tolist()} length={len(reader)}")
    return reader.get_batch(indices.astype(np.int64)).asnumpy()


def _instance_masks(video: Path, indices: np.ndarray) -> np.ndarray:
    """Return [T,H,W] foreground mask from the paired id annotations."""
    stem = video.name[:-8]
    with video.with_name(stem + ".pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    colors = np.asarray(metadata["static"].get("video_object_segmentation_colors", []), dtype=np.float32)
    annotation_path = video.with_name(stem + "_id.json")
    annotations = json.loads(annotation_path.read_text())
    if not len(colors):
        raise ValueError(f"no segmentation colors in {video}")
    result = []
    for frame_index in indices:
        frame_items = annotations.get(f"{int(frame_index):04d}", [])
        mask = None
        for item in frame_items:
            decoded = decode_rle(item)
            mask = decoded if mask is None else (mask | decoded)
        if mask is None:
            raise ValueError(f"missing segmentation annotation for frame {frame_index} in {annotation_path}")
        result.append(mask)
    return np.stack(result)


def _object_masks(video: Path, indices: np.ndarray, object_index: int) -> np.ndarray:
    """Return one segmentation object's [T,H,W] mask."""
    stem = video.name[:-8]
    annotation_path = video.with_name(stem + "_id.json")
    annotations = json.loads(annotation_path.read_text())
    result = []
    for frame_index in indices:
        item = next((x for x in annotations.get(f"{int(frame_index):04d}", []) if int(x["idx"]) == int(object_index)), None)
        if item is None:
            result.append(None)
        else:
            result.append(decode_rle(item))
    # Missing annotations are valid for temporarily occluded objects; infer
    # the frame shape from any available mask and represent them as empty.
    shape = next((x.shape for x in result if x is not None), None)
    if shape is None:
        raise ValueError(f"object idx={object_index} has no masks in {annotation_path}")
    return np.stack([x if x is not None else np.zeros(shape, dtype=bool) for x in result])


def _metadata(video: Path) -> dict:
    with video.with_name(video.name[:-8] + ".pkl").open("rb") as handle:
        return pickle.load(handle)


def _paste_object(recipient: np.ndarray, donor: np.ndarray, donor_mask: np.ndarray) -> np.ndarray:
    """Composite donor foreground onto recipient at its donor-frame location."""
    if recipient.shape[1:3] != donor.shape[1:3]:
        raise ValueError(f"recipient/donor resolution differs: {recipient.shape} vs {donor.shape}")
    result = recipient.copy()
    for t, mask in enumerate(donor_mask):
        result[t][mask] = donor[t][mask]
    return result


def _edit_frames(frames: np.ndarray, foreground: np.ndarray, mode: str) -> np.ndarray:
    edited = frames.copy()
    if mode == "base":
        return edited
    if mode == "background_replace":
        # Neutral gray avoids introducing a semantic object or motion cue.
        edited[~foreground] = np.asarray([128, 128, 128], dtype=np.uint8)
        return edited
    if mode == "horizontal_flip":
        return edited[:, :, ::-1].copy()
    if mode == "vertical_flip":
        return edited[:, ::-1, :].copy()
    raise ValueError(f"unknown edit mode: {mode}")


def _expected_targets(row: dict, mode: str, added: dict | None = None) -> dict:
    out = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in row.items()}
    state = np.array(row["state_2d"], copy=True)
    valid = np.asarray(row["state_valid"], dtype=bool)
    if mode == "horizontal_flip":
        state[..., 0] = 1.0 - state[..., 0]
        state[..., 5] *= -1.0
    elif mode == "vertical_flip":
        state[..., 1] = 1.0 - state[..., 1]
        state[..., 6] *= -1.0
    elif mode not in ("base", "background_replace"):
        if mode != "add_object":
            raise ValueError(mode)
    if mode == "add_object" and added is not None:
        free = np.flatnonzero(~np.asarray(out["object_present"], dtype=bool))
        if len(free):
            slot = int(free[0])
            out["object_present"][slot] = True
            out["object_type"][slot] = int(added["object_type"])
            out["color_rgb"][slot] = np.asarray(added.get("color_rgb", [0, 0, 0]), dtype=np.float32)
    out["state_2d"] = state
    out["state_valid"] = valid
    if mode == "add_object" and added is not None and len(free):
            state[slot] = np.asarray(added["state_2d"], dtype=np.float32)
            valid[slot] = np.asarray(added["state_valid"], dtype=bool)
    return out


def _probe_summary(output: dict, row: dict, assignment: torch.Tensor, vocab: list[str], pair_indices: torch.Tensor) -> dict:
    present = output["presence_logits"].sigmoid().cpu()
    type_prob = output["object_type_logits"].softmax(-1).cpu()
    summary = {"slots": [], "pairs": []}
    for slot in range(present.numel()):
        gt = int(assignment[slot]) if int(assignment[slot]) >= 0 else None
        gt_type = None
        if gt is not None and "object_type" in row:
            gt_id = int(torch.as_tensor(row["object_type"])[gt])
            if 0 <= gt_id < len(vocab):
                gt_type = vocab[gt_id]
        item = {
            "slot": slot,
            "presence_probability": float(present[slot]),
            "predicted_type": vocab[int(type_prob[slot].argmax())] if vocab else None,
            "ground_truth_type": gt_type,
            "type_correct": gt_type is not None and vocab[int(type_prob[slot].argmax())] == gt_type,
            "target_probability": float(output["target_logits"][slot].sigmoid().cpu()),
            "color_rgb": output["color_pred"][slot].cpu(),
            "ground_truth_object": gt,
            "trajectory_2d": output["trajectory_2d"][slot].cpu(),
        }
        summary["slots"].append(item)
    distance = output["pair_distance_2d"]
    contact = output["contact_logits"].sigmoid()
    first_contact = output["first_contact_logits"].softmax(-1)
    for index, (left, right) in enumerate(pair_indices.tolist()):
        summary["pairs"].append({
            "pair_index": index,
            "slots": [int(left), int(right)],
            "distance": distance[index],
            "contact_probability": contact[index],
            "first_contact_distribution": first_contact[index],
        })
    return summary


def _metrics(output: dict, row: dict, assignment: torch.Tensor) -> dict:
    pred_present = output["presence_logits"].sigmoid().ge(.5).cpu()
    matched = assignment.ge(0).cpu()
    type_pred = output["object_type_logits"].argmax(-1).cpu()
    type_true = torch.as_tensor(row["object_type"]).long()
    type_ok = []
    center, geometry, velocity = [], [], []
    for slot, target_index in enumerate(assignment.tolist()):
        if target_index < 0:
            continue
        type_ok.append(bool(type_pred[slot] == type_true[target_index]))
        valid = torch.as_tensor(row["state_valid"][target_index]).bool()
        if valid.any():
            p, t = output["trajectory_2d"][slot].cpu(), torch.as_tensor(row["state_2d"][target_index]).float()
            center.append(float((p[valid, :2] - t[valid, :2]).abs().mean()))
            geometry.append(float((p[valid, 2:5] - t[valid, 2:5]).abs().mean()))
            velocity.append(float((p[valid, 5:7] - t[valid, 5:7]).abs().mean()))
    return {
        "presence_f1": float(2 * (pred_present & matched).sum() / (2 * (pred_present & matched).sum() + (pred_present & ~matched).sum() + (~pred_present & matched).sum()).clamp_min(1)),
        "object_type_accuracy": float(np.mean(type_ok)) if type_ok else None,
        "center_mae": float(np.mean(center)) if center else None,
        "geometry_mae": float(np.mean(geometry)) if geometry else None,
        "velocity_mae": float(np.mean(velocity)) if velocity else None,
        "pair_distance_mean": float(output["pair_distance_2d"].mean().cpu()),
        "contact_probability_mean": float(output["contact_logits"].sigmoid().mean().cpu()),
    }


def _delta(value, base):
    if isinstance(value, dict) and isinstance(base, dict):
        return {k: _delta(value[k], base[k]) for k in value if k in base}
    if isinstance(value, (int, float)) and isinstance(base, (int, float)):
        return float(value - base)
    return None


def _tensor_deltas(outputs: dict, base: dict) -> dict:
    result = {}
    for key, value in outputs.items():
        if key not in base or not torch.is_tensor(value) or not torch.is_tensor(base[key]):
            continue
        diff = (value.float() - base[key].float()).abs()
        result[key] = {"mean_abs": float(diff.mean()), "max_abs": float(diff.max())}
    return result


def _equivariance(output: dict, base: dict, mode: str) -> dict:
    if mode not in ("horizontal_flip", "vertical_flip"):
        return {}
    pred, reference = output["trajectory_2d"].float(), base["trajectory_2d"].float()
    expected = reference.clone()
    axis, velocity = (0, 5) if mode == "horizontal_flip" else (1, 6)
    expected[..., axis] = 1.0 - expected[..., axis]
    expected[..., velocity] *= -1.0
    return {
        "trajectory_transformed_base_mae": float((pred - expected).abs().mean()),
        "position_transformed_base_mae": float((pred[..., :2] - expected[..., :2]).abs().mean()),
        "velocity_transformed_base_mae": float((pred[..., 5:7] - expected[..., 5:7]).abs().mean()),
    }


def _markdown_report(metadata: dict, metrics: dict, comparison: dict) -> str:
    lines = [
        "# Physion++ 单 clip 视频干预总结",
        "",
        f"- Seed：`{metadata['seed']}`",
        f"- Split：`{metadata['split']}`",
        f"- Base clip：`{metadata['video']}`",
        f"- Donor clip：`{metadata.get('donor_video', 'N/A')}`",
        f"- 添加物体：`{metadata.get('donor_object_type', 'N/A')}`（donor object index `{metadata.get('donor_object_index', 'N/A')}`）",
        "",
        "## 各版本核心指标",
        "",
        "| 版本 | Presence F1 | 类别准确率 | Center MAE | Geometry MAE | Velocity MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, item in metrics.items():
        lines.append(
            f"| `{mode}` | {item['presence_f1']:.3f} | {item['object_type_accuracy']:.3f} | "
            f"{item['center_mae']:.4f} | {item['geometry_mae']:.4f} | {item['velocity_mae']:.4f} |"
        )
    lines += ["", "## 物体类别识别", ""]
    for mode, item in metrics.items():
        lines += [f"### `{mode}`", "", "| Slot | 对应真实物体 | 预测类别 | 真实类别 | 是否正确 | Presence |", "|---:|---:|---|---|:---:|---:|"]
        for slot in item["probe_summary"]["slots"]:
            if slot["ground_truth_object"] is None and slot["presence_probability"] < 0.5:
                continue
            correct = "是" if slot["type_correct"] else "否"
            gt_type = slot["ground_truth_type"] or "—"
            gt_object = slot["ground_truth_object"] if slot["ground_truth_object"] is not None else "—"
            lines.append(f"| {slot['slot']} | {gt_object} | `{slot['predicted_type']}` | `{gt_type}` | {correct} | {slot['presence_probability']:.4f} |")
        lines.append("")
    lines += ["## 相对 base 的变化", "", "| 版本 | Presence F1 变化 | 类别准确率变化 | Center MAE 变化 | Geometry MAE 变化 |", "|---|---:|---:|---:|---:|"]
    for mode, item in comparison.items():
        delta = item["metric_delta_vs_base"]
        lines.append(f"| `{mode}` | {delta['presence_f1']:+.4f} | {delta['object_type_accuracy']:+.4f} | {delta['center_mae']:+.4f} | {delta['geometry_mae']:+.4f} |")
    lines += ["", "## 简要解读", ""]
    base = metrics["base"]
    if "add_object" in metrics:
        added = metrics["add_object"]
        if added["presence_f1"] < base["presence_f1"]:
            lines.append("- 添加物体后 probe 受到响应，但整体 Presence F1 下降，说明新增物体可能占用了 slot 或导致原有物体漏检。")
        else:
            lines.append("- 添加物体后 Presence F1 没有下降，需进一步检查新增物体是否获得独立且正确的 slot。")
    if "background_replace" in metrics:
        lines.append("- 背景替换主要用于检查物体语义和几何预测是否保持稳定。")
    if "horizontal_flip" in metrics and "vertical_flip" in metrics:
        lines.append("- 翻转版本应结合 `comparison.json` 中的 equivariance 误差判断坐标和速度是否按几何规则变化。")
    lines += ["", "原始完整 tensor、真实标签和详细 pair 输出保存在对应的 `*_probe.pt`、`ground_truth.json` 和 `probe_readable.json` 中。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("data_v1", "readout_data_v1", "testdata_v1"), default="readout_data_v1")
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-save-video", action="store_true", help="do not encode edited mp4 files")
    parser.add_argument("--no-save-probe", action="store_true", help="do not save raw *_probe.pt files")
    args = parser.parse_args()
    random.seed(args.seed)
    videos = sorted((args.dataset_root / args.split).glob("**/*_img.mp4"))
    video = args.video or random.Random(args.seed).choice(videos)
    if not video.is_file():
        raise FileNotFoundError(video)

    cfg = yaml.safe_load(args.config.read_text())["experiment"]
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    reader = VideoReader(str(video), num_threads=1, ctx=cpu(0))
    anchor = int(pickle.load(video.with_name(video.name[:-8] + ".pkl").open("rb"))["static"]["start_frame_for_prediction"])
    current_indices = anchor - 16 * int(data_cfg.get("current_frame_step", 2)) + np.arange(16) * int(data_cfg.get("current_frame_step", 2))
    future_indices = anchor + int(data_cfg.get("clip_gap", 0)) + np.arange(16) * int(data_cfg.get("future_frame_step", 4))
    all_indices = np.concatenate([current_indices, future_indices])
    frames = _read_frames(video, all_indices)
    foreground = _instance_masks(video, all_indices)
    # Read the clip's object names first.  The checkpoint vocabulary is loaded
    # below and takes precedence because logits are indexed in checkpoint order.
    names = set(["unknown"])
    with video.with_name(video.name[:-8] + ".pkl").open("rb") as handle:
        names.update(str(x.decode() if isinstance(x, bytes) else x) for x in pickle.load(handle)["static"].get("model_names", []))
    scene_vocab = sorted(names)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    predictor, _, _ = build_model(device, model_cfg, data_cfg, cfg["meta"]["pretrain_checkpoint"], cfg["meta"].get("encoder_checkpoint_key", "target_encoder"))
    predictor_payload = torch.load(args.predictor_checkpoint, map_location="cpu", weights_only=False)
    predictor.predictor.load_state_dict(predictor_payload["predictor"], strict=True)
    predictor.eval()
    probe_payload = torch.load(args.probe_checkpoint, map_location="cpu", weights_only=False)
    probe = PhysionDecoder(**probe_payload.get("model_config", {})).to(device)
    probe.load_state_dict(probe_payload["model"], strict=True)
    probe.eval()
    vocab = list(probe_payload.get("object_type_vocab", scene_vocab))
    # Keep the output readable even for older checkpoints that did not store a
    # vocabulary or stored fewer names than the classifier dimension.
    classifier_dim = int(probe.probe.object_type_head.out_features)
    if len(vocab) < classifier_dim:
        vocab.extend(f"class_{i}" for i in range(len(vocab), classifier_dim))
    target_row = targets(video, type_to_id={name: i for i, name in enumerate(vocab)})
    donor_candidates = [p for p in videos if p.resolve() != video.resolve()]
    donor = random.Random(args.seed + 1).choice(donor_candidates) if donor_candidates else video
    donor_meta = _metadata(donor)
    donor_names = [x.decode() if isinstance(x, bytes) else str(x) for x in donor_meta["static"].get("model_names", [])]
    donor_objects = list(range(min(len(donor_names), len(donor_meta["static"].get("video_object_segmentation_colors", donor_names)))))
    if not donor_objects:
        raise ValueError(f"donor has no segmentable objects: {donor}")
    donor_object = random.Random(args.seed + 2).choice(donor_objects)
    donor_anchor = int(donor_meta["static"]["start_frame_for_prediction"])
    donor_current = donor_anchor - 16 * int(data_cfg.get("current_frame_step", 2)) + np.arange(16) * int(data_cfg.get("current_frame_step", 2))
    donor_future = donor_anchor + int(data_cfg.get("clip_gap", 0)) + np.arange(16) * int(data_cfg.get("future_frame_step", 4))
    donor_indices = np.concatenate([donor_current, donor_future])
    donor_frames = _read_frames(donor, donor_indices)
    donor_masks = _object_masks(donor, donor_indices, donor_object)
    added_frames = _paste_object(frames, donor_frames, donor_masks)
    donor_type = donor_names[donor_object] if donor_object < len(donor_names) else "unknown"
    donor_row = targets(donor, type_to_id={name: i for i, name in enumerate(vocab)})
    added_state = np.asarray(donor_row["state_2d"][donor_object], dtype=np.float32)
    added_valid = donor_row["state_valid"][donor_object]
    added = {"object_type": vocab.index(donor_type) if donor_type in vocab else vocab.index("unknown"), "color_rgb": donor_row["color_rgb"][donor_object], "state_2d": added_state, "state_valid": added_valid}
    transform = make_transforms(False, (1., 1.), (1., 1.), 0., False, False, int(data_cfg.get("crop_size", 256)))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results, metrics, assignments = {}, {}, {}
    for mode in ("base", "background_replace", "horizontal_flip", "vertical_flip", "add_object"):
        edited = added_frames if mode == "add_object" else _edit_frames(frames, foreground, mode)
        # Save the edited source video when an ffmpeg-backed imageio writer is available.
        if not args.no_save_video:
            try:
                import imageio.v2 as imageio
                with imageio.get_writer(str(output_dir / f"{mode}.mp4"), fps=8) as writer:
                    for frame in edited:
                        writer.append_data(frame)
            except Exception as exc:
                (output_dir / f"{mode}.video_error.txt").write_text(str(exc) + "\n")
        tensor = transform(edited)
        current, future = tensor[:,:16].unsqueeze(0).to(device), tensor[:,16:].unsqueeze(0).to(device)
        with torch.inference_mode():
            context = predictor.encode_current(current)
            predicted = predictor.predict_next(context)
            out = probe(context.reshape(1, 8, 256, -1), predicted.reshape(1, 8, 256, -1))
        clean = {k: v[0].detach().cpu() for k, v in out.items() if torch.is_tensor(v)}
        expected = _expected_targets(target_row, mode, added if mode == "add_object" else None)
        assignment = match_objects(out, {"object_present": torch.as_tensor(expected["object_present"], device=device).unsqueeze(0), "state_2d": torch.as_tensor(expected["state_2d"], device=device).unsqueeze(0), "state_valid": torch.as_tensor(expected["state_valid"], device=device).unsqueeze(0)})[0].cpu()
        results[mode] = clean
        assignments[mode] = assignment
        metrics[mode] = _metrics(clean, expected, assignment)
        metrics[mode]["probe_summary"] = _probe_summary(clean, expected, assignment, vocab, probe.probe.pair_indices.cpu())
        if not args.no_save_probe:
            torch.save({"protocol": "physionpp3_single_clip_video_intervention_v1", "mode": mode, "path": str(video.resolve()), "current_indices": current_indices, "future_indices": future_indices, "outputs": clean, "assignment": assignment, "expected_targets": expected}, output_dir / f"{mode}_probe.pt")
    base_metrics = metrics["base"]
    comparison = {}
    for mode in metrics:
        if mode == "base":
            continue
        comparison[mode] = {
            "metric_delta_vs_base": _delta(metrics[mode], base_metrics),
            "tensor_delta_vs_base": _tensor_deltas(results[mode], results["base"]),
            "equivariance": _equivariance(results[mode], results["base"], mode),
        }
    (output_dir / "ground_truth.json").write_text(json.dumps({m: _expected_targets(target_row, m, added if m == "add_object" else None) for m in metrics}, default=_jsonable, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps({"metrics": metrics, "assignments": {k: v.tolist() for k, v in assignments.items()}}, default=_jsonable, indent=2) + "\n")
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    readable = {}
    for mode, item in metrics.items():
        readable[mode] = {
            "slots": [
                {k: s[k] for k in ("slot", "ground_truth_object", "predicted_type", "ground_truth_type", "type_correct", "presence_probability", "target_probability")}
                for s in item["probe_summary"]["slots"]
            ],
            "pairs": item["probe_summary"]["pairs"],
        }
    (output_dir / "probe_readable.json").write_text(json.dumps(readable, default=_jsonable, indent=2) + "\n")
    metadata = {"protocol": "physionpp3_single_clip_video_intervention_v1", "seed": args.seed, "video": str(video.resolve()), "split": args.split, "anchor": anchor, "current_indices": current_indices.tolist(), "future_indices": future_indices.tolist(), "donor_video": str(donor.resolve()), "donor_object_index": donor_object, "donor_object_type": donor_type, "probe_checkpoint": str(args.probe_checkpoint.resolve()), "predictor_checkpoint": str(args.predictor_checkpoint.resolve()), "config": str(args.config.resolve())}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "summary.md").write_text(_markdown_report(metadata, metrics, comparison), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "video": str(video.resolve()), "metrics": metrics, "comparison": comparison}, default=_jsonable, indent=2))


if __name__ == "__main__":
    main()
