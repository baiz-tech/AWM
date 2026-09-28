#!/usr/bin/env python3
"""Cache frozen EK100 encoder tokens at a configurable spatial resolution."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import torch,torch.distributed as dist,yaml
from torch.utils.data import DataLoader
from src.core.backbone import build_model
from src.data.ek100.dataset import EK100Dataset
from src.core.run_context import apply_cli_defaults, task_context

@torch.inference_mode()
def main():
 ctx=task_context();p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--annotations',required=True);p.add_argument('--indices',required=True);p.add_argument('--split',required=True);p.add_argument('--output',required=True);p.add_argument('--grid-size',type=int,default=8);p.add_argument('--batch-size',type=int,default=4);apply_cli_defaults(p,ctx);a=p.parse_args();assert 16%a.grid_size==0,'grid-size must divide the 16x16 encoder grid';cfg=yaml.safe_load(Path(a.config).read_text())
 if 'RANK' in os.environ:
  rank,world,local=map(int,(os.environ['RANK'],os.environ['WORLD_SIZE'],os.environ['LOCAL_RANK']));torch.cuda.set_device(local);dist.init_process_group('nccl');device=torch.device('cuda',local)
 else:rank,world,device=0,1,torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 index_payload=json.loads(Path(a.indices).read_text()); ids=index_payload[a.split][rank::world];d=cfg['data'];ds=EK100Dataset(a.annotations,d['video_root'],ids,int(d.get('crop_size',256)));dl=DataLoader(ds,a.batch_size,num_workers=int(d.get('num_workers',2)),pin_memory=True)
 m,_,_=build_model(device,cfg['predictor_model'],{'clip_frames':16,'crop_size':int(d.get('crop_size',256))},cfg['weights']['encoder'],cfg['weights'].get('encoder_key','target_encoder'));m.eval();xs=[];vs=[];ns=[];rec=[];block=16//a.grid_size
 for b in dl:
  count=len(b['verb_class']);z=m.encode_current(b['video'].to(device)).reshape(count,8,16,16,1280).reshape(count,8,a.grid_size,block,a.grid_size,block,1280).mean((3,5)).reshape(count,8,a.grid_size*a.grid_size,1280);xs.append(z.cpu().half());vs.append(b['verb_class']);ns.append(b['noun_class']);rec.extend([{k:b[k][i] for k in ('narration_id','participant_id','video_id','narration','verb','noun')} for i in range(count)])
 out=Path(a.output)
 if rank==0:out.mkdir(parents=True,exist_ok=False)
 if dist.is_initialized():dist.barrier()
 torch.save({'grid':torch.cat(xs),'verb':torch.cat(vs),'noun':torch.cat(ns),'records':rec},out/f'features-rank{rank:05d}.pt');n=torch.tensor([len(ds)],device=device)
 if dist.is_initialized():dist.all_reduce(n);dist.barrier()
 if rank==0:(out/'manifest.json').write_text(json.dumps({'split':a.split,'samples':int(n.item()),'shape':[8,a.grid_size*a.grid_size,1280]},indent=2)+'\n')
 if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
