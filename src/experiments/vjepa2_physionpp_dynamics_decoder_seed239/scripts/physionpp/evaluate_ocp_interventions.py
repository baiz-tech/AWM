#!/usr/bin/env python3
"""Evaluate OCP with visual/probe and slot-level probe interventions.

All conditions use the same cached latent samples and frozen checkpoints.  The
target slot is selected from ``is_target`` after the baseline Hungarian match;
the irrelevant slot is a non-target object with no target contact whenever
possible.  ``random_slot`` injects a copied valid object into one empty slot.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import PhysionDecoder, match_objects
from .ocp_readout import OCPReadout, VisualOnlyOCPReadout
from src.core.run_context import apply_cli_defaults, task_context


MODES = ("visual_only", "visual_probe", "target_mask", "irrelevant_mask", "random_slot")
PAIR_INDEX = {pair: i for i, pair in enumerate(itertools.combinations(range(8), 2))}


def cache_name(path: str | Path) -> str:
    return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:16] + ".pt"


class InterventionDataset(Dataset):
    def __init__(self, cache_root: Path, targets: Path, split: str, max_samples: int | None = None):
        payload = torch.load(targets, map_location="cpu", weights_only=False)
        self.targets = payload
        self.files = [Path(cache_root) / split / cache_name(path) for path in payload["path"]]
        if max_samples is not None:
            self.files = self.files[:max_samples]
            self.indices = list(range(max_samples))
        else:
            self.indices = list(range(len(self.files)))
        missing = [str(path) for path in self.files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing latent cache: {missing[0]}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        record = torch.load(self.files[index], map_location="cpu", weights_only=True)
        i = self.indices[index]
        keys = ("object_present", "state_2d", "state_valid", "is_target", "contact", "contact_valid")
        labels = {key: self.targets[key][i] for key in keys}
        return record["context_tokens"], record["future_tokens"], self.targets["ocp_label"][i], self.targets["ocp_valid"][i], labels


def binary_metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    scores, labels = scores.float(), labels.bool()
    pred = scores.sigmoid() >= threshold
    tp = int((pred & labels).sum()); tn = int((~pred & ~labels).sum())
    fp = int((pred & ~labels).sum()); fn = int((~pred & labels).sum())
    pos, neg, n = tp + fn, tn + fp, labels.numel()
    auroc = None
    if pos and neg:
        order = torch.argsort(scores)
        sorted_scores, sorted_labels = scores[order], labels[order]
        rank_sum = 0.0; start = 0
        while start < n:
            end = start + 1
            while end < n and sorted_scores[end] == sorted_scores[start]: end += 1
            rank_sum += ((start + 1 + end) / 2) * int(sorted_labels[start:end].sum()); start = end
        auroc = float((rank_sum - pos * (pos + 1) / 2) / (pos * neg))
    return {"samples": n, "positive": pos, "negative": neg, "threshold": threshold,
            "accuracy": (tp + tn) / n if n else None,
            "balanced_accuracy": ((tp / pos) + (tn / neg)) / 2 if pos and neg else None,
            "positive_recall": tp / pos if pos else None, "negative_recall": tn / neg if neg else None,
            "auroc": auroc, "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def mask_slot(probe: dict, slot: int) -> dict:
    out = {key: value.clone() for key, value in probe.items()}
    for key in ("object_tokens", "trajectory_2d"):
        out[key][slot] = 0
    for pair, index in PAIR_INDEX.items():
        if slot in pair:
            for key in ("pair_tokens", "contact_logits", "first_contact_logits"):
                out[key][index] = 0
    return out


def inject_random_slot(probe: dict, slot: int, source: int, generator: torch.Generator) -> dict:
    out = {key: value.clone() for key, value in probe.items()}
    # Copy a valid object's semantic/dynamic representation into an empty slot.
    out["object_tokens"][slot] = out["object_tokens"][source]
    out["trajectory_2d"][slot] = out["trajectory_2d"][source]
    for pair, index in PAIR_INDEX.items():
        if slot not in pair:
            continue
        other = pair[1] if pair[0] == slot else pair[0]
        # When the copied source is the same as this pair's other slot there is
        # no source self-pair (the probe only defines i < j pairs).  Keep the
        # injected slot's relation token neutral for this case.
        if source == other:
            for key in ("pair_tokens", "contact_logits", "first_contact_logits"):
                out[key][index] = 0
            continue
        source_pair = tuple(sorted((source, other)))
        source_index = PAIR_INDEX[source_pair]
        for key in ("pair_tokens", "contact_logits", "first_contact_logits"):
            out[key][index] = out[key][source_index]
    # Add a small deterministic perturbation so this is an intervention rather
    # than an exact duplicate under symmetric scenes.
    scale = out["object_tokens"].std().clamp_min(1e-6) * 0.01
    out["object_tokens"][slot] += torch.randn(out["object_tokens"][slot].shape, generator=generator) * scale
    return out


def choose_slots(output: dict, labels: dict) -> tuple[int | None, int | None, torch.Tensor]:
    device = output["presence_logits"].device
    batch = {key: value.to(device).unsqueeze(0) for key, value in labels.items() if key in ("object_present", "state_2d", "state_valid")}
    assignment = match_objects(output, batch)[0].detach().cpu()
    target_ids = torch.where(labels["is_target"].bool())[0].tolist()
    target_slot = next((slot for slot, obj in enumerate(assignment.tolist()) if obj in target_ids), None)
    candidates = []
    for slot, obj in enumerate(assignment.tolist()):
        if obj < 0 or obj in target_ids:
            continue
        contact = labels["contact"][target_ids[0], obj] if target_ids else torch.tensor(0)
        valid = labels["contact_valid"][target_ids[0], obj] if target_ids else torch.tensor(False)
        if not bool((contact.bool() & valid.bool()).any()):
            candidates.append(slot)
    irrelevant_slot = candidates[0] if candidates else next((slot for slot, obj in enumerate(assignment.tolist()) if obj >= 0 and slot != target_slot), None)
    return target_slot, irrelevant_slot, assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split", default="readout_data_v1")
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-probe-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--device", default="cuda:0")
    apply_cli_defaults(parser, task_context())
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", 0)); world_size = int(os.environ.get("WORLD_SIZE", 1)); local_rank = int(os.environ.get("LOCAL_RANK", rank))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed OCP intervention evaluation requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
    random.seed(args.seed + rank); torch.manual_seed(args.seed + rank); device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dataset = InterventionDataset(args.cache_root, args.targets, args.split, args.max_samples)
    if distributed:
        dataset.files = dataset.files[rank::world_size]
        dataset.indices = dataset.indices[rank::world_size]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    probe_payload = torch.load(args.probe_checkpoint, map_location="cpu", weights_only=False)
    probe = PhysionDecoder(**probe_payload.get("model_config", {})).to(device); probe.load_state_dict(probe_payload["model"], strict=True); probe.eval()
    visual_payload = torch.load(args.visual_checkpoint, map_location="cpu", weights_only=False)
    visual = VisualOnlyOCPReadout(**visual_payload.get("model_config", {})).to(device); visual.load_state_dict(visual_payload["model"], strict=True); visual.eval()
    vp_payload = torch.load(args.visual_probe_checkpoint, map_location="cpu", weights_only=False)
    visual_probe = OCPReadout(**vp_payload.get("model_config", {})).to(device); visual_probe.load_state_dict(vp_payload["model"], strict=True); visual_probe.eval()
    scores = {mode: [] for mode in MODES}; labels_all = []; deltas = {mode: [] for mode in MODES[1:]}; records = []
    generator = torch.Generator().manual_seed(args.seed)
    try:
      with torch.inference_mode():
        for context, future, label, valid, batch_labels in loader:
            use = valid.bool()
            if not use.any(): continue
            context, future = context.to(device).float(), future.to(device).float()
            out = probe(context, future)
            base_probe = {key: value.detach() for key, value in out.items() if torch.is_tensor(value)}
            base_score = visual_probe(torch.cat([context, future], 1), base_probe)
            visual_score = visual(torch.cat([context, future], 1))
            for row in range(context.size(0)):
                if not bool(use[row]): continue
                labels_all.append(label[row].detach().cpu()); scores["visual_only"].append(visual_score[row].detach().cpu()); scores["visual_probe"].append(base_score[row].detach().cpu())
                row_probe = {key: value[row].detach().cpu() for key, value in base_probe.items()}
                row_labels = {key: value[row].detach().cpu() for key, value in batch_labels.items()}
                target_slot, irrelevant_slot, assignment = choose_slots({key: value[row:row+1] for key, value in base_probe.items()}, row_labels)
                variants = {"target_mask": mask_slot(row_probe, target_slot) if target_slot is not None else row_probe, "irrelevant_mask": mask_slot(row_probe, irrelevant_slot) if irrelevant_slot is not None else row_probe}
                valid_slots = [slot for slot, obj in enumerate(assignment.tolist()) if obj >= 0]
                empty_slots = [slot for slot in range(8) if slot not in valid_slots]
                if empty_slots and valid_slots:
                    source = random.Random(args.seed + len(records)).choice(valid_slots); variants["random_slot"] = inject_random_slot(row_probe, empty_slots[0], source, generator)
                else:
                    variants["random_slot"] = row_probe
                item = {"target_slot": target_slot, "irrelevant_slot": irrelevant_slot, "assignment": assignment.tolist()}
                for mode, variant in variants.items():
                    score = visual_probe(torch.cat([context[row:row+1], future[row:row+1]], 1), {key: value.unsqueeze(0).to(device) for key, value in variant.items()})[0].detach().cpu(); scores[mode].append(score); deltas[mode].append(float(score - base_score[row].detach().cpu()))
                records.append(item)
    except Exception:
        if distributed:
            torch.distributed.destroy_process_group()
        raise
    local_payload = {"labels": [float(x) for x in labels_all], "scores": {mode: [float(x) for x in scores[mode]] for mode in MODES}, "deltas": deltas, "records": records}
    gathered = [None] * world_size if distributed and rank == 0 else None
    if distributed:
        torch.distributed.gather_object(local_payload, gathered, dst=0)
        torch.distributed.barrier()
        if rank != 0:
            torch.distributed.destroy_process_group()
            return
        payloads = gathered
    else:
        payloads = [local_payload]
    labels = torch.tensor(sum((x["labels"] for x in payloads), []), dtype=torch.float32)
    all_scores = {mode: torch.tensor(sum((x["scores"][mode] for x in payloads), []), dtype=torch.float32) for mode in MODES}
    all_deltas = {mode: sum((x["deltas"][mode] for x in payloads), []) for mode in MODES[1:]}
    all_records = sum((x["records"] for x in payloads), [])
    result = {"protocol": "physionpp3_ocp_intervention_v1", "split": args.split, "samples": len(labels), "world_size": world_size, "probe_checkpoint": str(args.probe_checkpoint.resolve()), "visual_checkpoint": str(args.visual_checkpoint.resolve()), "visual_probe_checkpoint": str(args.visual_probe_checkpoint.resolve()), "metrics": {mode: binary_metrics(all_scores[mode], labels) for mode in MODES}, "paired_delta_vs_visual_probe": {mode: {"mean_logit_delta": float(np.mean(all_deltas[mode])) if all_deltas[mode] else None, "mean_abs_logit_delta": float(np.mean(np.abs(all_deltas[mode]))) if all_deltas[mode] else None} for mode in MODES[1:]}, "records": all_records}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({k: result[k] for k in ("protocol", "samples", "metrics", "paired_delta_vs_visual_probe")}, indent=2))
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__": main()
