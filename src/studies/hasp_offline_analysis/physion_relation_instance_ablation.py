#!/usr/bin/env python3
"""Paired Physion++ relation-evidence intervention with predictor regeneration.

For each scene/window, real instance masks select one future-contact pair, one
same-scene non-contact pair, and one area-matched random pair.  Only context
patches are replaced; every intervention gets a fresh future latent from the
frozen predictor.  Baseline Hungarian assignment is reused for all variants.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu

from src.core.backbone import build_model
from src.experiments.awm_physionpp_fullpatch_probe_seed239.structured_probe import PhysionStructuredProbe
from .physion_input_ablation import summarize
from .physion_slot_time_probe import assignment
from src.core.run_context import apply_cli_defaults, task_context

PAIR_INDEX = {pair: i for i, pair in enumerate(itertools.combinations(range(8), 2))}


def object_patch_support(row: dict, *, clip_frames: int = 16, frame_step: int = 2) -> torch.Tensor:
    """Return [object,256] patch support using the exact cached current frames."""
    path = Path(row["path"])
    id_path = path.with_name(path.name.replace("_img.mp4", "_id.mp4"))
    pkl_path = path.with_name(path.name.replace("_img.mp4", ".pkl"))
    with pkl_path.open("rb") as handle:
        metadata = pickle.load(handle)
    colors = np.asarray(metadata["static"]["video_object_segmentation_colors"], dtype=np.float32)
    anchor = int(metadata["static"]["start_frame_for_prediction"])
    indices = anchor - clip_frames * frame_step + np.arange(clip_frames) * frame_step
    reader = VideoReader(str(id_path), num_threads=1, ctx=cpu(0))
    frames = reader.get_batch(indices.clip(0, len(reader) - 1)).asnumpy().astype(np.float32)
    dist = ((frames[..., None, :] - colors[None, None, None, :, :]) ** 2).sum(-1)
    label, nearest = dist.argmin(-1), dist.min(-1)
    label[nearest > 2500.0] = -1
    grid = label.reshape(clip_frames, 16, 16, 16, 16)
    support = np.zeros((8, 256), dtype=bool)
    for obj in range(min(8, len(colors))):
        support[obj] = (grid == obj).any((0, 2, 4)).reshape(-1)
    return torch.from_numpy(support)


def select_pairs(row: dict, support: torch.Tensor, seed: int = 239):
    """Select contact/non-contact/random pairs, matching random mask area."""
    valid, contact = row["future_contact_valid"].bool(), row["future_contact"].float()
    candidates = []
    for left, right in itertools.combinations(range(8), 2):
        if not valid[left, right].any():
            continue
        area = int((support[left] | support[right]).sum())
        candidates.append(((left, right), bool(contact[left, right].any()), area))
    contacts = [x for x in candidates if x[1] and x[2] > 0]
    noncontacts = [x for x in candidates if not x[1] and x[2] > 0]
    if not contacts or not noncontacts:
        return None
    target = max(contacts, key=lambda x: (x[2], -x[0][0], -x[0][1]))
    non = min(noncontacts, key=lambda x: (abs(x[2] - target[2]), x[0]))
    pool = [x for x in candidates if x[0] not in {target[0], non[0]}]
    if not pool:
        pool = noncontacts
    rng = np.random.default_rng(seed)
    # Random among the three closest areas, preserving a random-pair control.
    pool = sorted(pool, key=lambda x: abs(x[2] - target[2]))[: max(1, min(3, len(pool)))]
    random_pair = pool[int(rng.integers(len(pool)))]
    return {"contact": target[0], "noncontact": non[0], "random": random_pair[0],
            "areas": {"contact": target[2], "noncontact": non[2], "random": random_pair[2]}}


def mask_pair(context: torch.Tensor, support: torch.Tensor, pair: tuple[int, int] | None) -> torch.Tensor:
    if pair is None:
        return context.clone()
    selected = torch.where(support[pair[0]] | support[pair[1]])[0].to(context.device)
    result = context.clone()
    if len(selected):
        replacement = result.mean((1, 2), keepdim=True)
        result[:, :, selected] = replacement
    return result


def _pair_metrics(output, row, matched):
    contact_y, contact_s, distance, ttc = [], [], [], []
    for pair, pi in PAIR_INDEX.items():
        first, second = (int(matched[pair[0]]), int(matched[pair[1]]))
        if first < 0 or second < 0:
            continue
        cv = row["future_contact_valid"].bool()[first, second]
        if cv.any():
            contact_y.append(float((row["future_contact"][first, second][cv] > .5).any()))
            contact_s.append(float(output["contact_logits"][pi].sigmoid().amax()))
        dv = row["future_pair_valid"].bool()[first, second]
        if dv.any():
            distance.append(float((output["pair_distance"][pi][dv] - row["future_pair_distance"][first, second][dv]).abs().mean()))
        if bool(row["future_time_to_contact_valid"][first, second]):
            prob = output["first_contact_logits"][pi].softmax(-1)
            pred = (prob[:8] * torch.arange(8, dtype=prob.dtype)).sum() / 7
            ttc.append(float((pred - row["future_time_to_contact"][first, second]).abs()))
    return {"contact_y": contact_y, "contact_s": contact_s, "distance": distance, "ttc": ttc}


def _selected_contact_score(output, row, matched, pair):
    """Score the selected pair after baseline object assignment."""
    slots = {int(target): slot for slot, target in enumerate(matched.tolist()) if int(target) >= 0}
    if pair[0] not in slots or pair[1] not in slots:
        return None
    index = PAIR_INDEX[tuple(sorted((slots[pair[0]], slots[pair[1]])))]
    return float(output["contact_logits"][index].sigmoid().amax())


def _auroc(y, s):
    if len(y) == 0 or len(set(y)) < 2:
        return None
    y, s = np.asarray(y, bool), np.asarray(s, float)
    order = np.argsort(np.argsort(s)) + 1
    p, n = y.sum(), (~y).sum()
    return float((order[y].sum() - p * (p + 1) / 2) / (p * n))


def paired_bootstrap(values, seed=239, draws=2000):
    """95% percentile CI for scene-level paired deltas."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.asarray([values[rng.integers(len(values), size=len(values))].mean() for _ in range(draws)])
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], "n": int(len(values))}


def main():
    ctx = task_context()
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True); p.add_argument("--probe", required=True)
    p.add_argument("--predictor", required=True); p.add_argument("--predictor-config", required=True)
    p.add_argument("--output", required=True); p.add_argument("--max-samples", type=int)
    p.add_argument("--batch-size", type=int, default=2); p.add_argument("--device", default="cuda:0")
    apply_cli_defaults(p, ctx)
    a = p.parse_args(); device = torch.device(a.device)
    import yaml
    cfg = yaml.safe_load(Path(a.predictor_config).read_text())
    data_cfg = cfg["data"]
    predictor, _, _ = build_model(device, cfg["model"], data_cfg, cfg["meta"]["pretrain_checkpoint"], cfg["meta"].get("encoder_checkpoint_key", "target_encoder"))
    predictor.predictor.load_state_dict(torch.load(a.predictor, map_location="cpu", weights_only=False)["predictor"], strict=True); predictor.eval()
    probe = PhysionStructuredProbe().to(device)
    payload = torch.load(a.probe, map_location="cpu", weights_only=False); probe.load_state_dict(payload.get("model", payload), strict=True); probe.eval()
    files = sorted(Path(a.cache).glob("sample_*.pt")); files = files[:a.max_samples] if a.max_samples else files
    variants = ("baseline", "contact_pair", "noncontact_pair", "random_pair")
    results = {name: [] for name in variants}; metadata_rows = []; eligible = 0
    with torch.inference_mode():
        for start in range(0, len(files), a.batch_size):
            rows = [torch.load(path, map_location="cpu", weights_only=False) for path in files[start:start + a.batch_size]]
            selected = []
            for offset, row in enumerate(rows):
                support = object_patch_support(row, clip_frames=int(data_cfg.get("clip_frames", 16)), frame_step=int(data_cfg.get("current_frame_step", 2)))
                pair = select_pairs(row, support, seed=239 + start + offset)
                if pair is not None:
                    selected.append((row, support, pair)); metadata_rows.append({"path": row["path"], **pair})
            if not selected:
                continue
            eligible += len(selected)
            context = torch.stack([x[0]["context"] for x in selected]).float().to(device)
            cached = torch.stack([x[0]["future"] for x in selected]).float().to(device)
            base = probe(context, cached, return_features=True)
            assignments = [assignment({"trajectory_prediction": base["trajectory"][i].cpu(), "future_object_state": x[0]["future_object_state"], "future_object_valid": x[0]["future_object_valid"]}) for i, x in enumerate(selected)]
            variant_inputs = {"baseline": (context, cached)}
            for name, pair_key in (("contact_pair", "contact"), ("noncontact_pair", "noncontact"), ("random_pair", "random")):
                cx = torch.stack([mask_pair(context[i:i + 1], x[1], x[2][pair_key])[0] for i, x in enumerate(selected)])
                fy = predictor.predict_from_context(cx.reshape(cx.size(0), -1, cx.size(-1))).reshape_as(cached)
                variant_inputs[name] = (cx, fy)
            for name, (cx, fy) in variant_inputs.items():
                output = probe(cx, fy, return_features=True)
                for i, (row, _, pair) in enumerate(selected):
                    item = {"output": {k: v[i].detach().cpu() for k, v in output.items() if torch.is_tensor(v)}, "row": row, "assignment": assignments[i], "pair": pair}
                    item["pair_metrics"] = _pair_metrics(item["output"], row, assignments[i].cpu())
                    item["selected_contact_score"] = _selected_contact_score(item["output"], row, assignments[i].cpu(), pair[name.split("_")[0]] if name != "baseline" else pair["contact"])
                    item["all_pair_scores"] = {key: _selected_contact_score(item["output"], row, assignments[i].cpu(), pair[key]) for key in ("contact", "noncontact", "random")}
                    results[name].append(item)
    summaries = {name: summarize([x["output"] for x in items], [x["row"] for x in items], [x["assignment"] for x in items]) for name, items in results.items()}
    pair_summary = {}
    for name, items in results.items():
        pm = [x["pair_metrics"] for x in items]
        y, s = sum((x["contact_y"] for x in pm), []), sum((x["contact_s"] for x in pm), [])
        pair_summary[name] = {"contact_auroc": _auroc(y, s), "distance_mae": float(np.mean(sum((x["distance"] for x in pm), []))) if any(x["distance"] for x in pm) else None, "ttc_mae": float(np.mean(sum((x["ttc"] for x in pm), []))) if any(x["ttc"] for x in pm) else None}
    baseline = summaries["baseline"]
    delta = {name: {k: (None if summaries[name][k] is None or baseline[k] is None else summaries[name][k] - baseline[k]) for k in baseline} for name in variants if name != "baseline"}
    pair_delta = {name: {k: (None if pair_summary[name][k] is None or pair_summary["baseline"][k] is None else pair_summary[name][k] - pair_summary["baseline"][k]) for k in pair_summary["baseline"]} for name in variants if name != "baseline"}
    paired_selected = {}
    for name in variants:
        if name == "baseline":
            continue
        key = name.removesuffix("_pair")
        deltas = []
        for i, x in enumerate(results[name]):
            intervention_score = x["selected_contact_score"]
            baseline_score = results["baseline"][i]["all_pair_scores"][key]
            if intervention_score is not None and baseline_score is not None:
                deltas.append(intervention_score - baseline_score)
        paired_selected[name] = {"score_drop": paired_bootstrap([-v for v in deltas]), "raw_delta": paired_bootstrap(deltas)}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"protocol": "physion_true_instance_relation_pair_ablation_v2", "input_samples": len(files), "eligible_triplets": eligible, "summary": summaries, "delta_vs_baseline": delta, "pair_summary": pair_summary, "pair_delta_vs_baseline": pair_delta, "selected_contact_score_paired": paired_selected, "pair_selection": metadata_rows}
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x) + "\n")
    print(json.dumps({k: payload[k] for k in ("protocol", "input_samples", "eligible_triplets", "summary", "delta_vs_baseline", "pair_summary", "pair_delta_vs_baseline")}, indent=2))


if __name__ == "__main__":
    main()
