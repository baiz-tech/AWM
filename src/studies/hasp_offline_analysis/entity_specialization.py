"""Offline Entity-specialization metrics.

This module deliberately does not run an encoder or alter training.  It accepts
per-sample ``.pt`` rows exported by a probe and computes permutation-invariant
object matching, attribute accuracy, trajectory ADE/FDE, identity switches,
and EK100 noun-frequency/masking summaries.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Callable, Iterable

import torch
from src.core.run_context import apply_cli_defaults, task_context


def slot_object_matching(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor | None = None,
    presence_logits: torch.Tensor | None = None,
) -> list[tuple[int, int]]:
    """Return minimum-cost ``(pred_slot, target_object)`` pairs.

    ``predicted`` and ``target`` are ``[S,T,D]`` and ``[O,T,D]``.  The cost is
    validity-masked L1 over the first two channels (position), followed by all
    available channels.  Exhaustive permutations are intentional: Entity
    experiments use at most eight slots and this keeps the metric dependency
    free and exactly permutation invariant.
    """
    if predicted.ndim != 3 or target.ndim != 3:
        raise ValueError("predicted/target must be [slots, time, dims]")
    slots, objects = predicted.size(0), target.size(0)
    if not objects or not slots:
        return []
    valid = torch.ones(objects, target.size(1), dtype=torch.bool, device=target.device)
    if target_valid is not None:
        valid = target_valid.bool()
        if valid.shape != target.shape[:2]:
            raise ValueError("target_valid must be [objects, time]")
    diff = (predicted[:, None] - target[None]).abs()
    mask = valid[None, :, :, None].to(diff)
    costs = (diff * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
    if presence_logits is not None:
        costs = costs - 0.1 * presence_logits.sigmoid()[:, None]
    count = min(slots, objects)
    best: tuple[float, tuple[int, ...]] | None = None
    for chosen in itertools.permutations(range(slots), count):
        value = sum(float(costs[p, o]) for o, p in enumerate(chosen))
        if best is None or value < best[0]:
            best = (value, chosen)
    assert best is not None
    return [(int(p), int(o)) for o, p in enumerate(best[1])]


def ade_fde(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    """Compute mean ADE and FDE for matched ``[N,T,2+]`` trajectories."""
    if predicted.shape != target.shape or valid.shape != predicted.shape[:2]:
        raise ValueError("trajectory shapes must be [N,T,D], [N,T,D], [N,T]")
    error = torch.linalg.vector_norm(predicted[..., :2] - target[..., :2], dim=-1)
    values = error[valid.bool()]
    finals = []
    for row, mask in zip(error, valid.bool()):
        indices = mask.nonzero(as_tuple=False).flatten()
        if len(indices):
            finals.append(row[indices[-1]])
    return {
        "ade": float(values.mean()) if len(values) else float("nan"),
        "fde": float(torch.stack(finals).mean()) if finals else float("nan"),
        "points": int(values.numel()),
        "trajectories": len(finals),
    }


def identity_switch_rate(assignments: Iterable[torch.Tensor | list[int]]) -> dict[str, float | int]:
    """Measure slot identity switches in consecutive visible frames.

    Each assignment maps target-object index to predicted slot index; ``-1``
    denotes an occluded/unmatched object.  Transitions across an occlusion are
    counted as re-identification opportunities, not as switches by themselves.
    """
    previous: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    hidden: set[int] = set()
    opportunities = switches = reid_opportunities = reid_correct = 0
    for frame in assignments:
        current = {obj: int(slot) for obj, slot in enumerate(frame) if int(slot) >= 0}
        hidden.update(set(previous) - set(current))
        for obj, slot in current.items():
            if obj in previous:
                opportunities += 1
                switches += int(previous[obj] != slot)
            if obj in hidden:
                reid_opportunities += 1
                reid_correct += int(last_seen.get(obj) == slot)
            last_seen[obj] = slot
        previous = current
    return {"switches": switches, "opportunities": opportunities,
            "identity_switch_rate": switches / max(opportunities, 1),
            "reidentification_opportunities": reid_opportunities,
            "reidentification_accuracy": reid_correct / max(reid_opportunities, 1)}


def noun_frequency_buckets(records: Iterable[dict], bins: tuple[float, float] = (0.2, 0.8)) -> dict[int, str]:
    """Assign EK100 noun IDs to head/medium/tail using train frequencies."""
    counts: dict[int, int] = {}
    for row in records:
        noun = int(row["noun_class"] if "noun_class" in row else row["noun"])
        counts[noun] = counts.get(noun, 0) + 1
    ordered = sorted(counts, key=lambda noun: (-counts[noun], noun))
    n = len(ordered)
    first = max(1, int(n * bins[0])) if n else 0
    second = max(first + 1, int(n * bins[1])) if n > 1 else n
    second = min(second, n)
    return {noun: ("head" if i < first else "medium" if i < second else "tail") for i, noun in enumerate(ordered)}


def noun_bucket_metrics(
    labels: torch.Tensor, logits: torch.Tensor, buckets: dict[int, str], topk: int = 5
) -> dict[str, dict[str, float | int]]:
    """Report Top-1/Top-k by train-frequency bucket.

    Labels must use the same noun IDs as ``buckets`` and logits columns must be
    indexed by those IDs.  Missing bucket labels are reported as empty rather
    than silently folded into another bucket.
    """
    labels, logits = labels.long().cpu(), logits.float().cpu()
    prediction = logits.argmax(-1)
    top = logits.topk(min(topk, logits.size(-1)), dim=-1).indices
    result: dict[str, dict[str, float | int]] = {}
    for bucket in ("head", "medium", "tail"):
        keep = torch.tensor([buckets.get(int(label), "unknown") == bucket for label in labels], dtype=torch.bool)
        count = int(keep.sum())
        result[bucket] = {
            "count": count,
            "top1": float((prediction[keep] == labels[keep]).float().mean()) if count else float("nan"),
            f"top{topk}": float((top[keep] == labels[keep, None]).any(-1).float().mean()) if count else float("nan"),
        }
    return result


def masked_prediction_drop(
    grid: torch.Tensor,
    mask: torch.Tensor,
    predict: Callable[[torch.Tensor], torch.Tensor],
    labels: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate noun-score drop after masking the selected patch evidence.

    ``grid`` is ``[B,T,N,D]`` and ``mask`` is broadcastable to ``[B,T,N,1]``.
    The predictor must return noun logits ``[B,C]``.
    """
    baseline = predict(grid).detach()
    intervention = predict(grid * (~mask.bool()).to(grid).unsqueeze(-1)).detach()
    label = baseline.argmax(-1) if labels is None else labels.to(baseline.device).long()
    return {
        "baseline_top1": float((baseline.argmax(-1) == label).float().mean()),
        "masked_top1": float((intervention.argmax(-1) == label).float().mean()),
        "label_retention_drop": float((intervention.argmax(-1) != label).float().mean()),
    }


def _load_rows(root: str, limit: int | None) -> list[dict]:
    paths = sorted(Path(root).glob("sample_*.pt"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no sample_*.pt under {root}")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]


def evaluate_rows(rows: list[dict]) -> dict:
    """Evaluate rows containing prediction/target fields from an export."""
    matched_pred, matched_target, matched_valid = [], [], []
    switches = []
    attr_correct: dict[str, int] = {}
    attr_count: dict[str, int] = {}
    for row in rows:
        pred = row.get("predicted_trajectory", row.get("trajectory"))
        target = row.get("target_trajectory", row.get("future_object_state"))
        valid = row.get("target_valid", row.get("future_object_valid"))
        if pred is None or target is None or valid is None:
            continue
        pred, target, valid = pred.float(), target.float(), valid.bool()
        pairs = slot_object_matching(pred, target, valid, row.get("presence_logits"))
        if pairs:
            p, t = zip(*pairs)
            matched_pred.append(pred[list(p)]); matched_target.append(target[list(t)]); matched_valid.append(valid[list(t)])
            switches.append(torch.tensor([p for p, _ in pairs]))
        for name in ("color", "material", "shape"):
            logits, labels = row.get(f"{name}_logits"), row.get(f"target_{name}", row.get(name))
            if logits is not None and labels is not None:
                logits, labels = logits.float(), labels.long()
                if pairs and logits.ndim >= 2 and logits.size(0) >= max(p) + 1:
                    logits, labels = logits[list(p)], labels[list(t)]
                keep = labels >= 0
                attr_correct[name] = attr_correct.get(name, 0) + int((logits.argmax(-1)[keep] == labels[keep]).sum())
                attr_count[name] = attr_count.get(name, 0) + int(keep.sum())
    result: dict = {"protocol": "entity_specialization_metrics_v1", "samples": len(rows)}
    if matched_pred:
        result["trajectory"] = ade_fde(torch.cat(matched_pred), torch.cat(matched_target), torch.cat(matched_valid))
    result["attributes"] = {name: attr_correct[name] / max(attr_count[name], 1) for name in attr_correct}
    if switches:
        result["identity"] = identity_switch_rate(switches)
    return result


def main() -> None:
    ctx = task_context()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="directory containing exported sample_*.pt rows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args()
    result = evaluate_rows(_load_rows(args.features, args.max_samples))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
