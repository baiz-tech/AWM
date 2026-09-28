"""Cache VideoMAEv2 Current and Orca-aligned real-Future latents."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch, yaml
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
import numpy as np
from external.vjepa2.app.vjepa.transforms import make_transforms
from .model import build_encoder
from src.core.run_context import apply_cli_defaults, task_context

class PhysionWindows(Dataset):
    def __init__(self, root, split, frames=16, current_step=2, future_step=4, crop=224, max_videos=None):
        import glob, pickle
        self.frames=int(frames); self.cs=int(current_step); self.fs=int(future_step); self.transform=make_transforms(random_horizontal_flip=False,random_resize_aspect_ratio=(1.,1.),random_resize_scale=(1.,1.),crop_size=crop,normalize=((.5,.5,.5),(.5,.5,.5)))
        paths=sorted(glob.glob(str(Path(root)/split/'**/*_img.mp4'),recursive=True)); self.rows=[]
        if max_videos: paths=paths[:int(max_videos)]
        for path in paths:
            try:
                n=len(VideoReader(path,num_threads=1,ctx=cpu(0))); meta=pickle.load(open(path[:-8]+'.pkl','rb')); anchor=int(meta['static']['start_frame_for_prediction']); ci=anchor-self.frames*self.cs+np.arange(self.frames)*self.cs; fi=anchor+np.arange(self.frames)*self.fs
                # Align exactly with Orca: the canonical OCP label is the
                # metadata-level does_target_contact_zone field.
                label = int(bool(meta['static']['does_target_contact_zone']))
                if ci[0]>=0 and fi[-1]<n: self.rows.append((path,anchor,ci,fi,label))
            except (OSError,KeyError,TypeError,ValueError,EOFError,IndexError): continue
        if not self.rows: raise RuntimeError(f'no valid Physion++ samples in {split}')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        path,anchor,ci,fi,label=self.rows[i]; vr=VideoReader(path,num_threads=-1,ctx=cpu(0)); frames=vr.get_batch(np.concatenate((ci,fi))).asnumpy(); x=self.transform(frames); return {'current':x[:,:self.frames],'future':x[:,self.frames:],'path':path,'anchor_frame':anchor,'ocp_label':label}

def main():
    ctx=task_context();p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--split',required=True,choices=['data_v1','readout_data_v1','testdata_v1']); p.add_argument('--output-dir',required=True); p.add_argument('--batch-size',type=int); p.add_argument('--num-workers',type=int); p.add_argument('--max-videos',type=int); p.add_argument('--device',default='cuda'); p.add_argument('--rank',type=int,default=0); p.add_argument('--world-size',type=int,default=1); p.add_argument('--resume',action='store_true'); p.add_argument('--overwrite',action='store_true'); apply_cli_defaults(p,ctx);a=p.parse_args();
    if not 0 <= a.rank < a.world_size: raise ValueError('rank must satisfy 0 <= rank < world_size')
    cfg=yaml.safe_load(Path(a.config).read_text()); dcfg=cfg['data']; device=torch.device(a.device if torch.cuda.is_available() and a.device.startswith('cuda') else 'cpu'); enc=build_encoder(cfg['model']).to(device).eval(); ds=PhysionWindows(dcfg['root'],a.split,dcfg.get('clip_frames',16),dcfg.get('current_frame_step',2),dcfg.get('future_frame_step',4),dcfg.get('input_size',224),a.max_videos); ds.rows=ds.rows[a.rank::a.world_size]; dl=DataLoader(ds,batch_size=a.batch_size or dcfg.get('batch_size',2),shuffle=False,num_workers=a.num_workers if a.num_workers is not None else dcfg.get('num_workers',0),pin_memory=True); out=Path(a.output_dir)/a.split; out.mkdir(parents=True,exist_ok=True); manifest=[]
    with torch.no_grad():
      for b in dl:
        c=b['current'].to(device); f=b['future'].to(device); ct=enc(c).float().cpu(); ft=enc(f).float().cpu()
        for i,path in enumerate(b['path']):
          key=hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:16]; target=out/f'{key}.pt'; manifest.append(str(target))
          if a.resume and not a.overwrite and target.is_file(): continue
          torch.save({'protocol':'videomaev2_physionpp_orca_aligned_latent_v1','path':str(Path(path).resolve()),'anchor_frame':int(b['anchor_frame'][i]),'context_latent':ct[i].half(),'future_latent':ft[i].half(),'ocp_label':int(b['ocp_label'][i]),'ocp_valid':True},target)
    (out/f'manifest.rank{a.rank}.json').write_text(json.dumps({'protocol':'videomaev2_physionpp_orca_aligned_latent_v1','split':a.split,'rank':a.rank,'world_size':a.world_size,'samples':manifest},indent=2)+'\n')
if __name__=='__main__': main()
