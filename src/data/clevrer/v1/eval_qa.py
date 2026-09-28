#!/usr/bin/env python3
"""Evaluate ALOE-style CLEVRER QA or generate official test submissions."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from src.training.distributed import ExactDistributedSampler
from src.core.config_utils import merged_task_experiment
from src.data.clevrer.v1.qa_dataset import (
    ClevrerAloeTrajectoryDataset,
    WordTokenizer,
    clevrer_aloe_collate_fn,
    normalize_question_types,
)
from src.data.clevrer.v1.qa_model import build_qa_model
from src.data.clevrer.v1.qa_utils import (
    compute_metrics,
    gather_records,
    make_submission,
    records_from_outputs,
)
from src.data.clevrer.v1.protocol import resolve_qa_protocol


LOGGER = logging.getLogger("shared.evaluate.clevrer.eval_qa")


def experiment_config(raw_config):
    return merged_task_experiment(raw_config, "qa_eval")


def init_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def move_batch(batch, device):
    for key in (
        "scene_index",
        "question_id",
        "q_type",
        "q_subtype",
        "cls_video_emb",
        "cls_q_tokens",
        "cls_label",
        "mc_video_emb",
        "mc_subtype",
        "mc_q_tokens",
        "mc_label",
        "mc_flag",
        "mc_choice_id",
    ):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--trajectory-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=("validation", "val", "test"), default="validation")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--submission-only", action="store_true")
    args = parser.parse_args()

    split = "validation" if args.split == "val" else args.split
    if args.submission_only and split != "test":
        raise ValueError("--submission-only is only valid for the official test split")
    with open(args.config, encoding="utf-8") as handle:
        config = experiment_config(yaml.safe_load(handle))
    cfg_qa = config.get("qa", {})
    resolve_qa_protocol(config)
    cfg_outputs = config.get("outputs", {})
    rank, world_size, local_rank = init_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="[%(levelname)-8s][%(asctime)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    run_dir = Path(config["folder"])
    qa_run_dir = Path(cfg_outputs.get("qa_checkpoint_root", run_dir / "evaluations" / "qa_aloe"))
    checkpoint = Path(args.checkpoint or qa_run_dir / "best.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"QA checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_protocol = f"clevrer_{(payload.get('qa_config') or {}).get('backend', 'aloe_like_legacy')}_qa_v1"
    if payload.get("protocol") != expected_protocol:
        raise ValueError(
            f"incompatible QA checkpoint protocol: {payload.get('protocol')!r}; "
            f"expected {expected_protocol!r}"
        )
    tokenizer = WordTokenizer.from_state_dict(payload["tokenizer"])
    answers = payload["answer_vocabulary"]
    trajectory_root = Path(
        args.trajectory_root
        or cfg_outputs.get("qa_trajectory_root")
        or cfg_qa.get("trajectory_root", run_dir / "qa" / "trajectories")
    )
    annotation_root = Path(cfg_qa.get("annotation_root", "/data/shared/datasets/CLEVRER/official_code/executor/data"))
    question_types = normalize_question_types(payload.get("qa_config", {}).get("question_types", cfg_qa.get("question_types")))
    dataset = ClevrerAloeTrajectoryDataset(
        annotation_root,
        trajectory_root,
        split,
        tokenizer,
        answers,
        max_scenes=args.max_scenes,
        require_labels=split != "test",
        random_start=False,
        question_types=question_types,
        max_visual_tokens=int(payload["qa_config"].get("max_visual_tokens", payload["qa_config"].get("max_n_objects", 7))),
        n_sample_frames=int(payload["qa_config"].get("n_sample_frames", 25)),
        video_len=int(payload["qa_config"].get("video_len", 128)),
        protocol=str(payload["qa_config"].get("protocol", "compatibility_stride2")),
        frame_offset=int(
            (payload["qa_config"].get("temporal") or {}).get(
                "frame_offset",
                payload["qa_config"].get("video_len", 128)
                // payload["qa_config"].get("n_sample_frames", 25),
            )
        ),
        predictive_window=bool(
            (payload["qa_config"].get("temporal") or {}).get("predictive_window", True)
        ),
    )
    sampler = ExactDistributedSampler(dataset, rank, world_size)
    num_workers = int(cfg_qa.get("num_workers", 4))
    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=int(cfg_qa.get("val_batch_size", cfg_qa.get("train_batch_size", 192))),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
        collate_fn=clevrer_aloe_collate_fn,
    )
    model = build_qa_model(payload["qa_config"], payload["visual_dim"], tokenizer, answers).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    dtype_name = str(config.get("meta", {}).get("dtype", "bfloat16")).lower()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    use_amp = dtype_name in ("bfloat16", "float16") and device.type == "cuda"
    threshold = float(payload.get("mc_threshold", 0.5))
    records = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp, dtype=dtype if use_amp else None
            ):
                outputs = model(batch)
            records.extend(records_from_outputs(outputs, batch, threshold=threshold))
    records = gather_records(records)
    if rank == 0:
        output_dir = Path(args.output_dir or run_dir / "evaluations" / f"qa_aloe_{split}")
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "testfile" if split == "test" else "valfile"
        prediction_json = output_dir / f"{checkpoint.stem}_{suffix}.json"
        annotation_file = annotation_root / ("test.json" if split == "test" else "validation.json")
        scene_indices = [int(scene["scene_index"]) for scene in dataset.scenes]
        submission = make_submission(
            annotation_file,
            records,
            answers,
            scene_indices=scene_indices,
            question_types=question_types,
        )
        prediction_json.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
        if split == "test":
            (output_dir / "submission.json").write_text(
                json.dumps(submission, indent=2) + "\n", encoding="utf-8"
            )
            LOGGER.info("official test has no labels; wrote submission: %s", prediction_json)
        else:
            metrics = compute_metrics(records)
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (output_dir / "predictions.json").write_text(
                json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            LOGGER.info("validation metrics=%s", metrics)
        manifest = {
            "protocol": payload["protocol"],
            "split": split,
            "samples": len(dataset),
            "qa_checkpoint": str(checkpoint.resolve()),
            "qa_checkpoint_epoch": int(payload["epoch"]),
            "mc_threshold": threshold,
            "trajectory_root": str(trajectory_root.resolve()),
            "prediction_json": str(prediction_json.resolve()),
            "official_test_has_labels": False,
            "question_types": question_types,
            "world_size": world_size,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
