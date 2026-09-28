#!/usr/bin/env python3
"""Cache frozen naive-V-JEPA context/prediction tokens with Physion++ targets."""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import torch, yaml
import torch.distributed as dist
from torch.utils.data import DataLoader
from external.vjepa2.app.vjepa.transforms import make_transforms
from src.data.physionpp.dataset import PhysionCurrentFutureDataset
from src.core.backbone import build_model
from src.data.physionpp.targets import make_targets
from src.core.run_context import apply_cli_defaults, task_context
from .output_guard import ensure_no_artifacts


class Dataset(PhysionCurrentFutureDataset):
    def _load(self, index):
        sample = super()._load(index)
        if sample is None: return None
        with open(self._pkl_path(sample["path"]), "rb") as handle: metadata = pickle.load(handle)
        sample.update({key: torch.from_numpy(value) for key, value in make_targets(metadata, sample["current_indices"].numpy(), sample["future_indices"].numpy(), max_objects=8, tubelets=8).items()})
        return sample


@torch.inference_mode()
def main():
    ctx = task_context()
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--split", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--max-videos", type=int); parser.add_argument("--device", default="cuda:0")
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text()); data, model_cfg, meta = config["data"], config["model"], config["meta"]
    output = Path(args.output_dir)
    distributed = "RANK" in __import__("os").environ
    if distributed:
        local_rank = int(__import__("os").environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else: rank, world_size, local_rank = 0, 1, 0
    # The launcher writes configs and logs here before the module starts.
    ensure_no_artifacts(output)
    if distributed: dist.barrier()
    output.mkdir(parents=True, exist_ok=True); device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    transform = make_transforms(False, (1., 1.), (1., 1.), 0., False, False, int(data["crop_size"]))
    dataset = Dataset(root=data["root"], split=args.split, video_glob=data["video_glob"], clip_frames=data["clip_frames"], sampling_mode="prediction_start", current_frame_step=data["current_frame_step"], future_frame_step=data["future_frame_step"], clip_gap=data["clip_gap"], transform=transform, deterministic=True, max_videos=args.max_videos)
    indices = list(range(rank, len(dataset), world_size)); loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=int(data.get("num_workers", 2)), pin_memory=device.type == "cuda")
    world, _, _ = build_model(device, model_cfg, data, meta["pretrain_checkpoint"], meta.get("encoder_checkpoint_key", "target_encoder")); payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    world.predictor.load_state_dict(payload["predictor"], strict=True); world.eval()
    for index, batch in zip(indices, loader):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            context = world.encode_current(batch["current"].to(device))
            future = world.predict_from_context(context)
        record = {"context": context.reshape(8, 256, 1280).half().cpu(), "future": future.reshape(8, 256, 1280).half().cpu(), "path": batch["path"][0], **{key: value[0].cpu() for key, value in batch.items() if key.startswith(("current_object", "future_object", "future_pair", "future_contact", "future_time", "ocp_"))}}
        torch.save(record, output / f"sample_{index:06d}.pt")
    if distributed: dist.barrier()
    if rank == 0:
        manifest = {"protocol": "physionpp_fullpatch_structured_probe_v1", "split": args.split, "samples": len(dataset), "context_shape": [8, 256, 1280], "future_shape": [8, 256, 1280], "checkpoint": str(Path(args.checkpoint).resolve()), "world_size": world_size}
        (output / "cache_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if distributed: dist.destroy_process_group()

if __name__ == "__main__": main()
