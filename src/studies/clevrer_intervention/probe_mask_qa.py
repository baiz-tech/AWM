#!/usr/bin/env python3
"""Single-scene probe object masking intervention for CLEVRER predictive QA."""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import shutil
from pathlib import Path

import torch

from src.studies.clevrer_intervention.random_predictive_qa import (
    DEFAULT_ANNOTATIONS, DEFAULT_PROBE_CHECKPOINT, DEFAULT_QA_CHECKPOINT,
    DEFAULT_TRAJECTORY_ROOT, DEFAULT_VIDEO_ROOT, _load_json, _qa_batch,
    choose_question, load_probe,
)
from src.data.clevrer.qa_dataset import WordTokenizer
from src.data.clevrer.qa_model import build_qa_model

COLORS = ("gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow")
MATERIALS = ("rubber", "metal")
SHAPES = ("cube", "cylinder", "sphere")


def descriptions(question):
    text = " ".join([question["question"]] + [c["choice"] for c in question["choices"]]).lower()
    found = []
    for shape in SHAPES:
        if re.search(rf"\b{shape}\b", text): found.append((shape, {"shape": shape}))
    for color in COLORS:
        if re.search(rf"\b{color}\b", text): found.append((color, {"color": color}))
    for material in MATERIALS:
        if re.search(rf"\b{material}\b", text): found.append((material, {"material": material}))
    return found


def match_slots(probe, descs):
    valid = probe["object_valid"].bool()
    scores = []
    for text, attrs in descs:
        score = torch.zeros(probe["presence_logits"].shape[0])
        for attr, value in attrs.items():
            names = {"color": COLORS, "material": MATERIALS, "shape": SHAPES}[attr]
            logits = probe[f"{attr}_logits"]
            score += logits[:, names.index(value)].float().log_softmax(-1)
        score[~valid] = -float("inf")
        scores.append((text, attrs, int(score.argmax()), float(score.max())))
    return scores


def mask_probe(probe, slot):
    out = {k: v.clone() for k, v in probe.items()}
    for key in ("object_tokens", "presence_logits", "color_logits",
                "material_logits", "shape_logits", "trajectory_2d"):
        out[key][slot] = 0
    # QA's structured encoder requires at least one valid object pair. If the
    # scene has only two valid objects, keep the masked slot structurally valid
    # while zeroing all its content; with >=3 objects it can be invalidated.
    valid_count = int(probe["object_valid"].sum().item())
    if valid_count >= 3:
        out["object_valid"][slot] = False
    pair_indices = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    for index, (i, j) in enumerate(pair_indices):
        if slot in (i, j):
            for key in ("pair_tokens", "pair_distance_2d", "contact_gt_event", "first_contact_logits"):
                out[key][index] = 0
            if valid_count >= 3:
                out["pair_valid"][index] = False
    return out


def infer(model, tokenizer, question, video, probe, device):
    batch = _qa_batch(question, tokenizer, video, probe)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch["mc_probe_outputs"] = {k: v.to(device) for k, v in batch["mc_probe_outputs"].items()}
    with torch.inference_mode(): logits = model(batch)["mc_answer_logits"].float().cpu()
    probs = logits.sigmoid().tolist(); pred = int(logits.argmax())
    choices = [{"choice_id": int(c["choice_id"]), "choice": c["choice"],
                "ground_truth": c.get("answer"), "probability": float(p),
                "predicted": i == pred} for i, (c, p) in enumerate(zip(question["choices"], probs))]
    correct = [i for i, c in enumerate(question["choices"]) if c.get("answer") == "correct"]
    binary_predictions = [p >= 0.5 for p in probs]
    labels = [c.get("answer") == "correct" for c in question["choices"]]
    return {"choices": choices, "predicted_answer": choices[pred]["choice"],
            "ground_truth_answer": [choices[i]["choice"] for i in correct],
            "is_correct": pred in correct,
            "option_predictions": binary_predictions,
            "official_question_correct": binary_predictions == labels,
            "official_option_correct": [p == y for p, y in zip(binary_predictions, labels)]}


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scene, question = choose_question(args.annotations, args.trajectory_root, args.video_root, args.seed, args.scene_id)
    scene_id = int(scene["scene_index"]); question = dict(question); question["_scene_index"] = scene_id
    trajectory = args.trajectory_root / "validation" / f"scene_{scene_id:05d}.pt"
    video = args.video_root / f"video_{scene_id:05d}.mp4"
    record = torch.load(trajectory, map_location="cpu", weights_only=True)
    visual = record["visual_tokens"].float()
    probe_model, probe_payload = load_probe(args.probe_checkpoint, device)
    with torch.inference_mode(): raw = probe_model(visual[:8].to(device)[None], visual[8:].to(device)[None])
    valid = raw["presence_logits"].sigmoid().ge(0.5)
    first, second = probe_model.probes.pair_indices[:, 0], probe_model.probes.pair_indices[:, 1]
    probe = {"object_tokens": raw["object_tokens"][0].cpu(), "pair_tokens": raw["pair_tokens"][0].cpu(),
             "object_valid": valid[0].cpu(), "pair_valid": (valid[:, first] & valid[:, second])[0].cpu()}
    for name in ("presence_logits", "color_logits", "material_logits", "shape_logits", "trajectory_2d", "pair_distance_2d", "contact_gt_event", "first_contact_logits"):
        probe[name] = raw[name][0].float().cpu()
    qa = torch.load(args.qa_checkpoint, map_location="cpu", weights_only=False)
    tokenizer = WordTokenizer.from_state_dict(qa["tokenizer"])
    model = build_qa_model(qa["qa_config"], qa["visual_dim"], tokenizer, qa["answer_vocabulary"]).to(device)
    model.load_state_dict(qa["model"], strict=True); model.eval()
    matches = match_slots(probe, descriptions(question))
    relevant = random.Random(args.seed).choice(matches)
    mentioned = {m[2] for m in matches}; candidates = [i for i, x in enumerate(probe["object_valid"]) if x and i not in mentioned]
    if not candidates: candidates = [i for i, x in enumerate(probe["object_valid"]) if x and i != relevant[2]]
    if not candidates: raise ValueError("no irrelevant valid object slot available")
    irrelevant = random.Random(args.seed + 1).choice(candidates)
    baseline = infer(model, tokenizer, question, visual, probe, device)
    rel = infer(model, tokenizer, question, visual, mask_probe(probe, relevant[2]), device)
    irr = infer(model, tokenizer, question, visual, mask_probe(probe, irrelevant), device)
    out = args.output_root / f"seed_{args.seed:03d}"; out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video, out / video.name)
    result = {"protocol": "clevrer_probe_mask_qa_v1", "seed": args.seed, "scene_index": scene_id,
              "question_id": int(question["question_id"]), "question": question["question"], "choices": question["choices"],
              "relevant": {"description": relevant[0], "slot": relevant[2], "score": relevant[3]},
              "irrelevant": {"slot": irrelevant}, "runs": {"baseline": baseline, "relevant_mask": rel, "irrelevant_mask": irr},
              "video": str(video.resolve()), "video_output": str((out / video.name).resolve()),
              "trajectory": str(trajectory.resolve()), "probe_checkpoint": str(args.probe_checkpoint.resolve()),
              "qa_checkpoint": str(args.qa_checkpoint.resolve())}
    (out / "probe_mask_qa.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=239); p.add_argument("--scene-id", type=int, default=None)
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS); p.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    p.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT); p.add_argument("--probe-checkpoint", type=Path, default=DEFAULT_PROBE_CHECKPOINT)
    p.add_argument("--qa-checkpoint", type=Path, default=DEFAULT_QA_CHECKPOINT); p.add_argument("--output-root", type=Path, default=Path("outputs/runs/reproduce_v1_clevrer_only/clevrer_analyseIntervention/probe_mask_qa")); p.add_argument("--device", default="cuda:0")
    run(p.parse_args())


if __name__ == "__main__": main()
