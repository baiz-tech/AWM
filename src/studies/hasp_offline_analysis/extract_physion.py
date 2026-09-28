#!/usr/bin/env python3
"""Export compact HASP states from the existing Physion++ cache.

The encoder/predictor cache is reused and the trained structured probe is
frozen.  Each output file contains entity, dynamic, relation and readout
features plus the physical targets needed by offline probes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import flatten_feature, save_json
from src.experiments.awm_physionpp_fullpatch_probe_seed239.structured_probe import PhysionStructuredProbe
from src.core.run_context import apply_cli_defaults, task_context


def main() -> None:
    ctx = task_context()
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    apply_cli_defaults(p, ctx)
    a = p.parse_args()
    files = sorted(Path(a.cache).glob("sample_*.pt"))
    if a.max_samples:
        files = files[: a.max_samples]
    files = files[a.rank::a.world_size]
    if not files:
        raise FileNotFoundError(f"no sample_*.pt under {a.cache}")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    model = PhysionStructuredProbe().to(device)
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model", payload), strict=True)
    model.eval()
    rows = []
    with torch.inference_mode():
        for index, path in enumerate(files):
            sample = torch.load(path, map_location="cpu", weights_only=False)
            context = sample["context"].unsqueeze(0).to(device)
            future = sample["future"].unsqueeze(0).to(device)
            result = model(context, future, return_features=True)
            row = {
                "object_tokens": result["object_tokens"][0].cpu().half(),
                "time_features": result["time_features"][0].cpu().half(),
                "pair_tokens": result["pair_tokens"][0].cpu().half(),
                "pair_time_features": result["pair_time_features"][0].cpu().half(),
                "trajectory_prediction": result["trajectory"][0].cpu().half(),
                "presence_logits": result["presence_logits"][0].cpu().float(),
                "entity": flatten_feature(result["object_tokens"])[0].cpu().half(),
                "dynamic": flatten_feature(result["time_features"])[0].cpu().half(),
                "relation": flatten_feature(result["pair_tokens"])[0].cpu().half(),
                "full": torch.cat((flatten_feature(result["object_tokens"]),
                                    flatten_feature(result["time_features"]),
                                    flatten_feature(result["pair_tokens"])), -1)[0].cpu().half(),
                "ocp_label": sample["ocp_label"], "ocp_valid": sample["ocp_valid"],
                "future_object_state": sample["future_object_state"],
                "future_object_valid": sample["future_object_valid"],
                "future_pair_distance": sample["future_pair_distance"],
                "future_pair_valid": sample["future_pair_valid"],
                "future_contact": sample["future_contact"],
                "future_contact_valid": sample["future_contact_valid"],
                "future_time_to_contact": sample["future_time_to_contact"],
                "future_time_to_contact_valid": sample["future_time_to_contact_valid"],
                "path": sample.get("path", ""),
            }
            global_index = index * a.world_size + a.rank
            torch.save(row, out / f"sample_{global_index:06d}.pt")
            rows.append({"index": global_index, "source": str(path), "path": sample.get("path", "")})
    if a.rank == 0:
        save_json(out / "manifest.json", {"protocol": "hasp_physion_offline_features_v1", "count": len(rows) * a.world_size, "features": ["entity", "dynamic", "relation", "full"], "source": str(Path(a.cache).resolve()), "probe": str(Path(a.checkpoint).resolve()), "world_size": a.world_size})


if __name__ == "__main__":
    main()
