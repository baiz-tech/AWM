from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import yaml

from .model_variants import AttributionExtractor, AttributionHead, VARIANTS, load_components
from .train import GridDataset
from src.data.ek100.dataset import read_rows
from src.data.ek100.labels import encode_labels
from torch.utils.data import DataLoader
from src.core.run_context import apply_cli_defaults, task_context


@torch.no_grad()
def evaluate_variant(name, extractor, head, loader, device, action_map):
    values = {k: [] for k in ("verb_top1", "verb_top5", "noun_top1", "noun_top5", "action_top1")}
    for x, v, n in loader:
        x, v, n = x.to(device), v.to(device), n.to(device); target = torch.tensor([action_map.get((int(a), int(b)), -1) for a, b in zip(v.cpu(), n.cpu())], device=device); verb, noun, action = head(extractor(x)); valid = target >= 0
        values["verb_top1"].extend((verb.argmax(1) == v).cpu().tolist()); values["verb_top5"].extend((verb.topk(min(5, verb.size(1)), 1).indices == v[:, None]).any(1).cpu().tolist()); values["noun_top1"].extend((noun.argmax(1) == n).cpu().tolist()); values["noun_top5"].extend((noun.topk(min(5, noun.size(1)), 1).indices == n[:, None]).any(1).cpu().tolist())
        if valid.any(): values["action_top1"].extend((action[valid].argmax(1) == target[valid]).cpu().tolist())
    return {k: (sum(v) / len(v) if v else None) for k, v in values.items()} | {"samples": len(values["verb_top1"])}


def main():
    ctx=task_context();p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--validation", required=True); p.add_argument("--readout", required=True); p.add_argument("--adapter", required=True); p.add_argument("--checkpoints", required=True, help="directory holding the five heads, as <name>.pt or <name>/best.pt"); p.add_argument("--output", required=True); p.add_argument("--device", default="cuda"); apply_cli_defaults(p,ctx);a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text()); rows = read_rows(cfg["data"]["train_annotations"]); vv = sorted({int(r["verb_class"]) for r in rows}); nv = sorted({int(r["noun_class"]) for r in rows}); pairs = sorted({(vv.index(int(r["verb_class"])), nv.index(int(r["noun_class"]))) for r in rows}); action_map = {pair: i for i, pair in enumerate(pairs)}
    checkpoints = Path(a.checkpoints)
    for name in VARIANTS:
        flat = checkpoints / f"{name}.pt"
        nested = checkpoints / name / "best.pt"
        if not flat.is_file() and not nested.is_file():
            raise FileNotFoundError(
                f"missing {name} head: looked for {flat} and {nested}. Train the "
                f"variant first (the train_* tasks write <experiment_root>/<name>/best.pt)."
            )
    ds = GridDataset(a.validation); ds.v, ds.n = encode_labels(ds.v, vv), encode_labels(ds.n, nv); loader = DataLoader(ds, 64); device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"); comp = load_components(a.readout, a.adapter, device); result = {"protocol": "ek100_fair_attribution_e0_e4_v1", "variants": {}, "head_parameter_count": None}
    for name in VARIANTS:
        flat = checkpoints / f"{name}.pt"
        source = flat if flat.is_file() else checkpoints / name / "best.pt"
        head = AttributionHead(len(vv), len(nv), pairs).to(device); payload = torch.load(source, map_location="cpu", weights_only=False); head.load_state_dict(payload["model"], strict=True); ex = AttributionExtractor(name, comp.readout_world, comp.shared).to(device); result["variants"][name] = evaluate_variant(name, ex, head, loader, device, action_map); result["head_parameter_count"] = sum(p.numel() for p in head.parameters())
    base = result["variants"]["E0"]
    result["delta_vs_E0"] = {name: {k: (vals[k] - base[k] if vals[k] is not None and base[k] is not None else None) for k in ("verb_top1", "noun_top1", "action_top1")} for name, vals in result["variants"].items()}
    result["checkpoints"] = str(checkpoints.resolve())
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
