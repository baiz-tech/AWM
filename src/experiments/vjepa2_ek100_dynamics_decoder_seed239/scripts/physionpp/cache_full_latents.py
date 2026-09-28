#!/usr/bin/env python3
"""Cache frozen V-JEPA encoder context and native predictor future latents."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import torch,yaml
from decord import VideoReader,cpu
from torch.utils.data import DataLoader
from external.vjepa2.app.vjepa.transforms import make_transforms
from src.experiments.vjepa2_clevrer_naive_world_model_seed239.model import build_model
from src.experiments.vjepa2_clevrer_naive_world_model_seed239.dataset import PhysionCurrentFutureDataset

def cache_name(path):
 return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:16]+'.pt'

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--predictor-checkpoint',type=Path,required=True);p.add_argument('--split',choices=('data_v1','readout_data_v1','testdata_v1'),required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--num-workers',type=int,default=2);p.add_argument('--max-videos',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args();cfg=yaml.safe_load(a.config.read_text());exp=cfg['experiment'];meta,mc,dc=exp['meta'],exp['model'],exp['data'];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model,_,_=build_model(device,mc,dc,meta['pretrain_checkpoint'],meta.get('encoder_checkpoint_key','target_encoder'));payload=torch.load(a.predictor_checkpoint,map_location='cpu',weights_only=False)
 if payload.get('protocol')!='current_16_to_future_16':raise ValueError(f"unexpected predictor protocol: {payload.get('protocol')!r}")
 model.predictor.load_state_dict(payload['predictor'],strict=True);model.eval();[x.requires_grad_(False) for x in model.parameters()];transform=make_transforms(False,(1.,1.),(1.,1.),0.,False,False,int(dc.get('crop_size',256)));ds=PhysionCurrentFutureDataset(root=dc['root'],split=a.split,video_glob=dc.get('video_glob','**/*_img.mp4'),clip_frames=16,sampling_mode='prediction_start',current_frame_step=2,future_frame_step=4,clip_gap=0,transform=transform,deterministic=True,max_videos=a.max_videos);loader=DataLoader(ds,batch_size=a.batch_size,num_workers=a.num_workers,pin_memory=device.type=='cuda');out=a.output_dir/a.split;out.mkdir(parents=True,exist_ok=True);written=0
 with torch.no_grad():
  for batch in loader:
   current=batch['current'].to(device);context=model.encode_current(current);future=model.predict_next(context);context=context.reshape(-1,8,256,1280).half().cpu();future=future.reshape(-1,8,256,1280).half().cpu()
   for i,path in enumerate(batch['path']):
    target=out/cache_name(path)
    if a.resume and target.is_file():continue
    torch.save({'protocol':'physionpp_fullpatch_native_predictor_latent_v1','path':str(path),'current_indices':batch['current_indices'][i],'future_indices':batch['future_indices'][i],'context_tokens':context[i],'future_tokens':future[i]},target);written+=1
 manifest={'protocol':'physionpp_fullpatch_native_predictor_latent_v1','split':a.split,'samples':len(list(out.glob('*.pt'))),'context_shape':[8,256,1280],'future_shape':[8,256,1280],'predictor_checkpoint':str(a.predictor_checkpoint.resolve())};(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'written':written,**manifest},indent=2))
if __name__=='__main__':main()
