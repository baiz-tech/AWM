from __future__ import annotations
import argparse,json,os
from pathlib import Path
import torch, yaml, torch.distributed as dist
from torch.utils.data import DataLoader
from .dataset import EK100Dataset
from .model import build_encoder
from src.core.run_context import apply_cli_defaults, task_context

@torch.no_grad()
def main():
 ctx=task_context();p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--annotations',required=True); p.add_argument('--indices'); p.add_argument('--split',required=True); p.add_argument('--output',required=True); p.add_argument('--batch-size',type=int,default=2); apply_cli_defaults(p,ctx);a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); rank=int(os.getenv('RANK',0)); world=int(os.getenv('WORLD_SIZE',1)); local=int(os.getenv('LOCAL_RANK',0)); dev=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu');
 if world>1: torch.cuda.set_device(local); dist.init_process_group('nccl')
 ids=None
 if a.indices: ids=json.loads(Path(a.indices).read_text())[a.split][rank::world]
 ds=EK100Dataset(a.annotations,cfg['data']['video_root'],ids,cfg['data'].get('crop_size',224)); dl=DataLoader(ds,a.batch_size,num_workers=cfg['data'].get('num_workers',2),pin_memory=True); enc=build_encoder(cfg['model'],dev); xs=[]; vs=[]; ns=[]
 for batch_index, b in enumerate(dl, 1):
  z=enc.patch_embed(b['video'].to(dev)); pos=enc.pos_embed.to(z); z=z+pos
  for blk in enc.blocks: z=blk(z)
  # Match official VideoMAEv2 forward_features: mean over all tokens first,
  # then apply fc_norm (the previous order changed the representation).
  z=enc.fc_norm(z.mean(1)).cpu().half(); xs.append(z); vs.append(torch.tensor(b['verb'])); ns.append(torch.tensor(b['noun']))
  if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(dl):
   print(f'cache progress rank={rank} split={a.split} batch={batch_index}/{len(dl)} samples={min(batch_index*a.batch_size, len(ds))}/{len(ds)}', flush=True)
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True); torch.save({'features':torch.cat(xs),'verb':torch.cat(vs),'noun':torch.cat(ns)},out/f'features-rank{rank:05d}.pt');
 if world>1: dist.barrier(); dist.destroy_process_group()
if __name__=='__main__': main()
