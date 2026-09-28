#!/usr/bin/env python3
"""Cache frozen Orca video-token mean latents for EK100."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch
from decord import VideoReader, cpu

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.core.orca_model import load_orca, extract_video_tokens
from src.core.run_context import apply_cli_defaults, task_context

def split_rows(rows, split, seed=239):
    if split == 'final_validation': return rows
    vids = sorted({r['video_id'] for r in rows})
    rng = np.random.default_rng(seed); rng.shuffle(vids)
    val = set(vids[:max(1, int(round(.1 * len(vids))))])
    return [r for r in rows if (r['video_id'] in val) == (split == 'probe_validation')]

def name(row):
    return hashlib.sha1(str(row['narration_id']).encode()).hexdigest()[:20] + '.pt'

def main():
    ctx = task_context()
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir', type=Path, default=Path('/data/shared_model/Orca-4B'))
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--video-root', type=Path, required=True)
    p.add_argument('--split', choices=['train','probe_validation','final_validation'], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--frames', type=int, default=16)
    p.add_argument('--prompt', default='Represent the observed action in this video.')
    p.add_argument('--seed', type=int, default=239)
    p.add_argument('--rank', type=int, default=int(os.getenv('RANK', 0)))
    p.add_argument('--world-size', type=int, default=int(os.getenv('WORLD_SIZE', 1)))
    p.add_argument('--max-samples', type=int)
    p.add_argument('--resume', action='store_true')
    apply_cli_defaults(p, ctx)
    a = p.parse_args()
    if not 0 <= a.rank < a.world_size: raise ValueError('invalid rank/world-size')
    rows = list(csv.DictReader(a.annotations.open(newline='')))
    rows = split_rows(rows, a.split, a.seed)
    if a.max_samples: rows = rows[:a.max_samples]
    rows = rows[a.rank::a.world_size]
    out = a.output_dir / a.split; out.mkdir(parents=True, exist_ok=True)
    local_rank = int(os.getenv('LOCAL_RANK', a.rank))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.set_device(local_rank)
    print(f'rank={a.rank} local_rank={local_rank} device={device} split={a.split} samples={len(rows)}', flush=True)
    model, processor = load_orca(a.checkpoint_dir, device)
    records = []; skipped = 0
    for i, row in enumerate(rows):
        target = out / name(row)
        if a.resume and target.exists(): records.append(str(target)); continue
        path = a.video_root / row['participant_id'] / 'videos' / f"{row['video_id']}.MP4"
        if not path.exists(): path = path.with_suffix('.mp4')
        if not path.exists(): raise FileNotFoundError(path)
        vr = VideoReader(str(path), num_threads=1, ctx=cpu(0)); n = len(vr)
        start = max(0, min(int(row['start_frame']) - 1, n - 1)); stop = max(start, min(int(row['stop_frame']) - 1, n - 1))
        indices = np.linspace(start, stop, a.frames).round().astype(np.int64)
        frames = vr.get_batch(indices).asnumpy()
        tokens, token_count = extract_video_tokens(model, processor, frames, a.prompt, device)
        torch.save({'protocol':'orca_ek100_frozen_tokens_v1','video_tokens':tokens.half(),'verb':int(row['verb_class']),
                    'noun':int(row['noun_class']),'video_id':row['video_id'],'narration_id':row['narration_id'],
                    'frame_indices':indices.tolist(),'video_token_count':token_count}, target)
        records.append(str(target))
        if i == 0 or (i + 1) % 10 == 0: print(f'rank={a.rank} split={a.split} cached={i+1}/{len(rows)}', flush=True)
    (a.output_dir / f'manifest_{a.split}.rank{a.rank}.json').write_text(json.dumps({'protocol':'orca_ek100_frozen_latent_v1','split':a.split,'rank':a.rank,'world_size':a.world_size,'samples':len(records),'skipped':skipped,'paths':records}, indent=2) + '\n')

if __name__ == '__main__': main()
