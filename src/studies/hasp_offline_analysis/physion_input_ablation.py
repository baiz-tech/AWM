#!/usr/bin/env python3
"""Input-evidence ablations for frozen Physion++ HASP.

No Entity/Dynamic/Relation decoder is masked.  We perturb cached patch latent
evidence, rerun the frozen structured probe, and compare native readouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.experiments.awm_physionpp_fullpatch_probe_seed239.structured_probe import PhysionStructuredProbe
from .physion_slot_time_probe import assignment
from src.core.run_context import apply_cli_defaults, task_context


def auroc(y, score):
    y, score = np.asarray(y), np.asarray(score)
    if len(np.unique(y)) != 2: return None
    order = np.argsort(np.argsort(score)) + 1; positive = y.astype(bool); p, n = positive.sum(), (~positive).sum()
    return float((order[positive].sum() - p * (p + 1) / 2) / (p * n)) if p and n else None


def mask_object_evidence(context, future, attention, ratio=0.25):
    """Mask top-attended spatial patches for each object slot."""
    b, slots, memory = attention.shape
    context = context.clone(); future = future.clone()
    # memory order is 8 context frames followed by 8 future frames.
    for sample in range(b):
        score = attention[sample].reshape(slots, 16, 256)
        flat = score[:, :8].mean(1).amax(0)
        k = max(1, int(round(256 * ratio)))
        selected = flat.topk(k, dim=-1).indices
        context[sample, :, selected] = context[sample, :, selected].mean((0, 1), keepdim=True)
        future[sample, :, selected] = future[sample, :, selected].mean((0, 1), keepdim=True)
    return context, future


def mask_random_evidence(context, future, ratio=0.25, seed=239):
    context = context.clone(); future = future.clone(); g = torch.Generator(device=context.device).manual_seed(seed)
    k = max(1, int(round(256 * ratio))); selected = torch.randperm(256, generator=g, device=context.device)[:k]
    context[:, :, selected] = context[:, :, selected].mean((0, 1), keepdim=True)
    future[:, :, selected] = future[:, :, selected].mean((0, 1), keepdim=True)
    return context, future


def temporal_static(context, future):
    value = torch.cat((context, future), 1).mean(1, keepdim=True)
    return value.expand_as(context).clone(), value.expand_as(future).clone()


def temporal_shuffle(context, future, seed=239):
    g = torch.Generator(device=context.device).manual_seed(seed)
    order = torch.randperm(8, generator=g, device=context.device)
    return context[:, order], future[:, order]


def relation_swap(context, future, attention, presence_logits, ratio=0.25):
    """Swap the two strongest distinct slot supports in spatial latent space."""
    context, future = context.clone(), future.clone()
    score = attention[:, :, :8 * 256].reshape(attention.size(0), 8, 8, 256).mean(2)
    for sample in range(context.size(0)):
        first, second = presence_logits[sample].topk(2).indices.tolist()
        k = max(8, int(round(256 * ratio)))
        a = score[sample, first].topk(k).indices
        candidate = score[sample, second].argsort(descending=True)
        b = candidate[~torch.isin(candidate, a)][:len(a)]
        if len(b) < len(a):
            a = a[:len(b)]
        temp = context[sample, :, a].clone(); context[sample, :, a] = context[sample, :, b]; context[sample, :, b] = temp
        temp = future[sample, :, a].clone(); future[sample, :, a] = future[sample, :, b]; future[sample, :, b] = temp
    return context, future


def relation_layout_shuffle(context, future, seed=239):
    """Permute 4x4 spatial blocks while preserving all local patch content."""
    g = torch.Generator(device=context.device).manual_seed(seed)
    blocks = torch.randperm(16, generator=g, device=context.device)
    def permute(value):
        b, t, _, d = value.shape
        grid = value.reshape(b, t, 16, 16, d).reshape(b, t, 4, 4, 4, 4, d)
        tiles = grid.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, t, 16, 4, 4, d)
        tiles = tiles[:, :, blocks]
        return tiles.reshape(b, t, 4, 4, 4, 4, d).permute(0, 1, 2, 4, 3, 5, 6).reshape(b, t, 256, d)
    return permute(context), permute(future)


def summarize(outputs, rows, assignments):
    presence_y, presence_s, contact_y, contact_s, ocp_y, ocp_s = [], [], [], [], [], []
    extent_error, speed_error, distance_error, ttc_error = [], [], [], []
    for out, row, assignment_value in zip(outputs, rows, assignments):
        # Use the frozen baseline slot-to-object assignment for every
        # intervention. This keeps the comparison paired and prevents a
        # degraded intervention from changing the evaluation correspondence.
        matched = assignment_value.ge(0).float()
        presence_y.extend(matched.tolist()); presence_s.extend(out["presence_logits"].sigmoid().tolist())
        for slot, target_index in enumerate(assignment_value.tolist()):
            if target_index < 0: continue
            valid_time = row["future_object_valid"].bool()[target_index]
            if not valid_time.any(): continue
            prediction, target = out["trajectory"][slot][valid_time], row["future_object_state"].float()[target_index][valid_time]
            extent_error.append(float((prediction[:, 6:9] - target[:, 6:9]).abs().mean()))
            speed_error.append(float((prediction[:, 3:6].norm(dim=-1) - target[:, 3:6].norm(dim=-1)).abs().mean()))
        for index, (left, right) in enumerate(__import__("itertools").combinations(range(8), 2)):
            first, second = int(assignment_value[left]), int(assignment_value[right])
            if first < 0 or second < 0: continue
            valid = row["future_contact_valid"].bool()[first, second]
            if not valid.any(): continue
            target = (row["future_contact"].float()[first, second][valid] > .5).any()
            score = out["contact_logits"][index].sigmoid().amax()
            contact_y.append(float(target)); contact_s.append(float(score))
            distance_valid = row["future_pair_valid"].bool()[first, second]
            if distance_valid.any():
                distance_error.append(float((out["pair_distance"][index][distance_valid] - row["future_pair_distance"].float()[first, second][distance_valid]).abs().mean()))
            if bool(row["future_time_to_contact_valid"][first, second]):
                probabilities = out["first_contact_logits"][index].softmax(-1)
                predicted_ttc = (probabilities[:8] * torch.arange(8, dtype=probabilities.dtype)).sum() / 7
                ttc_error.append(float((predicted_ttc - row["future_time_to_contact"].float()[first, second]).abs()))
        if bool(row["ocp_valid"]): ocp_y.append(float(row["ocp_label"])); ocp_s.append(float(out["ocp_logits"].sigmoid()))
    return {"presence_auroc": auroc(presence_y, presence_s), "entity_extent_mae": float(np.mean(extent_error)), "dynamic_speed_mae": float(np.mean(speed_error)), "contact_auroc": auroc(contact_y, contact_s), "relation_distance_mae": float(np.mean(distance_error)), "relation_ttc_mae": float(np.mean(ttc_error)), "ocp_auroc": auroc(ocp_y, ocp_s) if len(set(ocp_y)) == 2 else None}


def main():
    ctx=task_context();p = argparse.ArgumentParser(); p.add_argument("--cache", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True); p.add_argument("--max-samples", type=int); p.add_argument("--device", default="cuda:0"); p.add_argument("--ratio", type=float, default=.25); apply_cli_defaults(p,ctx);a = p.parse_args()
    files = sorted(Path(a.cache).glob("sample_*.pt")); files = files[:a.max_samples] if a.max_samples else files
    device = torch.device(a.device); model = PhysionStructuredProbe().to(device); payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False); model.load_state_dict(payload.get("model", payload), strict=True); model.eval()
    all_results = {name: [] for name in ("baseline", "object_mask", "random_mask", "temporal_static", "temporal_shuffle", "relation_swap", "relation_layout_shuffle")}
    deltas = {name: [] for name in all_results if name != "baseline"}
    with torch.inference_mode():
        for path in files:
            row = torch.load(path, map_location="cpu", weights_only=False); context=row["context"].unsqueeze(0).to(device); future=row["future"].unsqueeze(0).to(device)
            base=model(context,future,return_features=True,return_attention=True); attention=base["object_memory_attention"]
            baseline_assignment = assignment({"trajectory_prediction": base["trajectory"][0].cpu(), "future_object_state": row["future_object_state"], "future_object_valid": row["future_object_valid"]})
            variants={"baseline":(context,future),"object_mask":mask_object_evidence(context,future,attention,a.ratio),"random_mask":mask_random_evidence(context,future,a.ratio),"temporal_static":temporal_static(context,future),"temporal_shuffle":temporal_shuffle(context,future),"relation_swap":relation_swap(context,future,attention,base["presence_logits"],a.ratio),"relation_layout_shuffle":relation_layout_shuffle(context,future)}
            for name,(x,y) in variants.items():
                out=model(x,y,return_features=True); all_results[name].append({"output":{k:v[0].detach().cpu() for k,v in out.items() if torch.is_tensor(v)},"row":row,"assignment":baseline_assignment})
    summary={}
    for name,items in all_results.items(): summary[name]=summarize([x["output"] for x in items],[x["row"] for x in items],[x["assignment"] for x in items])
    baseline=summary["baseline"]
    for name in deltas: deltas[name]={k:(None if summary[name][k] is None or baseline[k] is None else summary[name][k]-baseline[k]) for k in baseline}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);(out.with_suffix('.json')).write_text(json.dumps({"protocol":"physion_input_evidence_ablation_v1","samples":len(files),"mask_ratio":a.ratio,"summary":summary,"delta_vs_baseline":deltas},indent=2)+'\n');print(json.dumps({"summary":summary,"delta_vs_baseline":deltas},indent=2))

if __name__ == "__main__": main()
