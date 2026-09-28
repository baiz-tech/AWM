#!/usr/bin/env python3
"""Evaluate baseline/relevant/irrelevant probe masks on CLEVRER validation QA."""
from __future__ import annotations

import argparse
import json
import random
import os
from pathlib import Path

import torch

from src.studies.clevrer_intervention.probe_mask_qa import (
    descriptions, infer, load_probe, match_slots, mask_probe,
)
from src.studies.clevrer_intervention.random_predictive_qa import (
    DEFAULT_ANNOTATIONS, DEFAULT_PROBE_CHECKPOINT, DEFAULT_QA_CHECKPOINT,
    DEFAULT_TRAJECTORY_ROOT, DEFAULT_VIDEO_ROOT, _load_json,
)
from src.data.clevrer.qa_dataset import WordTokenizer
from src.data.clevrer.qa_model import build_qa_model
from src.core.run_context import apply_cli_defaults, task_context


def evaluate(args):
    distributed = "RANK" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun distributed evaluation requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scenes = _load_json(args.annotations / "validation.json")
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    scene_count = len(scenes)
    scenes = scenes[rank::world_size]
    probe_model, probe_payload = load_probe(args.probe_checkpoint, device)
    qa_payload = torch.load(args.qa_checkpoint, map_location="cpu", weights_only=False)
    tokenizer = WordTokenizer.from_state_dict(qa_payload["tokenizer"])
    qa_model = build_qa_model(
        qa_payload["qa_config"], qa_payload["visual_dim"], tokenizer,
        qa_payload["answer_vocabulary"],
    ).to(device)
    qa_model.load_state_dict(qa_payload["model"], strict=True)
    qa_model.eval()

    totals = {name: {"questions": 0, "question_correct": 0, "options": 0, "option_correct": 0}
              for name in ("baseline", "relevant_mask", "irrelevant_mask")}
    records = []
    for scene in scenes:
        scene_id = int(scene["scene_index"])
        trajectory = args.trajectory_root / "validation" / f"scene_{scene_id:05d}.pt"
        video = args.video_root / f"video_{scene_id:05d}.mp4"
        if not trajectory.is_file() or not video.is_file():
            raise FileNotFoundError(f"missing scene assets: trajectory={trajectory}, video={video}")
        record = torch.load(trajectory, map_location="cpu", weights_only=True)
        visual = record["visual_tokens"].float()
        with torch.inference_mode():
            raw = probe_model(visual[:8].to(device)[None], visual[8:].to(device)[None])
        valid = raw["presence_logits"].sigmoid().ge(0.5)
        first, second = probe_model.probes.pair_indices[:, 0], probe_model.probes.pair_indices[:, 1]
        probe = {
            "object_tokens": raw["object_tokens"][0].float().cpu(),
            "pair_tokens": raw["pair_tokens"][0].float().cpu(),
            "object_valid": valid[0].cpu(),
            "pair_valid": (valid[:, first] & valid[:, second])[0].cpu(),
        }
        for name in ("presence_logits", "color_logits", "material_logits", "shape_logits",
                     "trajectory_2d", "pair_distance_2d", "contact_gt_event", "first_contact_logits"):
            probe[name] = raw[name][0].float().cpu()
        for question in scene["questions"]:
            if question.get("question_type") != "predictive":
                continue
            q = dict(question); q["_scene_index"] = scene_id
            matches = match_slots(probe, descriptions(q))
            if not matches:
                raise ValueError(f"cannot identify objects from question {scene_id}/{question['question_id']}")
            rng = random.Random(args.seed + scene_id * 1009 + int(question["question_id"]))
            relevant = rng.choice(matches)
            mentioned = {item[2] for item in matches}
            candidates = [i for i, is_valid in enumerate(probe["object_valid"].tolist()) if is_valid and i not in mentioned]
            if not candidates:
                candidates = [i for i, is_valid in enumerate(probe["object_valid"].tolist()) if is_valid and i != relevant[2]]
            if not candidates:
                raise ValueError(f"no irrelevant object slot for {scene_id}/{question['question_id']}")
            irrelevant = rng.choice(candidates)
            runs = {
                "baseline": infer(qa_model, tokenizer, q, visual, probe, device),
                "relevant_mask": infer(qa_model, tokenizer, q, visual, mask_probe(probe, relevant[2]), device),
                "irrelevant_mask": infer(qa_model, tokenizer, q, visual, mask_probe(probe, irrelevant), device),
            }
            correct = {i for i, choice in enumerate(question["choices"]) if choice.get("answer") == "correct"}
            for name, result in runs.items():
                predicted = next(i for i, choice in enumerate(result["choices"]) if choice["predicted"])
                totals[name]["questions"] += 1
                totals[name]["question_correct"] += int(result["official_question_correct"])
                totals[name]["options"] += len(question["choices"])
                totals[name]["option_correct"] += sum(result["official_option_correct"])
            records.append({
                "scene_index": scene_id, "question_id": int(question["question_id"]),
                "question": question["question"], "choices": question["choices"],
                "relevant": {"description": relevant[0], "attributes": relevant[1], "slot": relevant[2], "score": relevant[3]},
                "irrelevant": {"slot": irrelevant}, "runs": runs,
            })
        if rank == 0 and len(records) and len(records) % 100 == 0:
            print(f"processed scenes through {scene_id}; questions={len(records)}", flush=True)

    if distributed:
        metric_tensor = torch.tensor(
            [[totals[name][key] for key in ("questions", "question_correct", "options", "option_correct")]
             for name in ("baseline", "relevant_mask", "irrelevant_mask")],
            dtype=torch.long, device=device,
        )
        torch.distributed.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
        for row, name in enumerate(("baseline", "relevant_mask", "irrelevant_mask")):
            for column, key in enumerate(("questions", "question_correct", "options", "option_correct")):
                totals[name][key] = int(metric_tensor[row, column].item())
        gathered_records = [None for _ in range(world_size)] if rank == 0 else None
        torch.distributed.gather_object(records, gathered_records, dst=0)
        if rank == 0:
            records = [item for chunk in gathered_records for item in chunk]
        torch.distributed.barrier()
    if rank != 0:
        return

    for result in totals.values():
        result["question_accuracy"] = result["question_correct"] / result["questions"] if result["questions"] else 0.0
        result["option_accuracy"] = result["option_correct"] / result["options"] if result["options"] else 0.0
    output = args.output or args.output_root / "validation_probe_mask_qa.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "clevrer_validation_probe_mask_qa_v1", "seed": args.seed,
        "scenes": scene_count, "predictive_questions": len(records),
        "metrics": totals, "records": records,
        "trajectory_root": str(args.trajectory_root.resolve()),
        "probe_checkpoint": str(args.probe_checkpoint.resolve()),
        "qa_checkpoint": str(args.qa_checkpoint.resolve()),
        "probe_epoch": int(probe_payload.get("epoch", -1)),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "metrics": totals}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--probe-checkpoint", type=Path, default=DEFAULT_PROBE_CHECKPOINT)
    parser.add_argument("--qa-checkpoint", type=Path, default=DEFAULT_QA_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs/reproduce_v1_clevrer_only/clevrer_analyseIntervention/probe_mask_qa_validation"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    apply_cli_defaults(parser, task_context())
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
