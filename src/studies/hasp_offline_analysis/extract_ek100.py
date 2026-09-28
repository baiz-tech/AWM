#!/usr/bin/env python3
"""Export frozen V23 HASP states from an existing EK100 patch cache."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import flatten_feature, save_json
from src.experiments.awm_ek100_multiscale_adapter_seed239.readout import EK100MultiScaleReadout
from src.experiments.awm_ek100_multiscale_adapter_seed239.model import SharedMultiScaleWorldExtractor
from src.core.run_context import apply_cli_defaults, task_context


def main() -> None:
    ctx = task_context()
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--readout", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    apply_cli_defaults(p, ctx)
    a = p.parse_args()
    paths = sorted(Path(a.cache).glob("features-rank*.pt"))
    if not paths:
        raise FileNotFoundError(f"no features-rank*.pt under {a.cache}")
    if a.world_size > 1:
        paths = [paths[a.rank]]
    shards = [torch.load(x, map_location="cpu", weights_only=False) for x in paths]
    grid = torch.cat([x["grid"] for x in shards]).float()
    verb, noun = torch.cat([x["verb"] for x in shards]), torch.cat([x["noun"] for x in shards])
    records = sum((x["records"] for x in shards), [])
    readout_payload = torch.load(a.readout, map_location="cpu", weights_only=False)
    action_map = readout_payload["action_map"]
    pairs = [None] * len(action_map)
    for pair, index in action_map.items():
        pairs[int(index)] = pair
    readout = EK100MultiScaleReadout(readout_payload["verb_classes"], readout_payload["noun_classes"], pairs)
    readout.load_state_dict(readout_payload["model"], strict=True)
    world = SharedMultiScaleWorldExtractor()
    adapter_payload = torch.load(a.adapter, map_location="cpu", weights_only=False)
    world.load_state_dict(adapter_payload["shared"], strict=True)
    readout.eval(); world.eval()
    count = min(len(grid), a.max_samples or len(grid))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device); readout.to(device); world.to(device)
    with torch.inference_mode():
        for index in range(count):
            x = grid[index:index + 1].to(device)
            shared_state, _ = world.world(x)
            refined = world(x)
            # The V23 classifier contains a second world extractor.  These are
            # the states actually consumed by its verb/noun/action heads and
            # are therefore the primary HASP analysis features.
            state, _ = readout.world(refined)
            verb_logits, noun_logits, action_logits, _ = readout(refined)
            row = {
                "entity": flatten_feature(state.objects)[0].cpu().half(),
                "dynamic": flatten_feature(state.dynamics)[0].cpu().half(),
                "relation": flatten_feature(state.relations)[0].cpu().half(),
                "scene": flatten_feature(state.scene)[0].cpu().half(),
                "full": refined.mean((1, 2))[0].cpu().half(),
                "shared_entity": flatten_feature(shared_state.objects)[0].cpu().half(),
                "shared_dynamic": flatten_feature(shared_state.dynamics)[0].cpu().half(),
                "shared_relation": flatten_feature(shared_state.relations)[0].cpu().half(),
                "verb": verb[index], "noun": noun[index],
                "verb_logits": verb_logits[0].cpu().float(), "noun_logits": noun_logits[0].cpu().float(),
                "action_logits": action_logits[0].cpu().float(),
                "record": records[index],
            }
            torch.save(row, out / f"sample_{a.rank:02d}_{index:06d}.pt")
    if a.rank == 0:
        save_json(out / "manifest.json", {"protocol": "hasp_ek100_offline_features_v2", "count": count * a.world_size, "features": ["entity", "dynamic", "relation", "scene", "full"], "auxiliary_features": ["shared_entity", "shared_dynamic", "shared_relation"], "source": str(Path(a.cache).resolve()), "readout": str(Path(a.readout).resolve()), "adapter": str(Path(a.adapter).resolve()), "world_size": a.world_size})


if __name__ == "__main__":
    main()
