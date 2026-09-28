#!/usr/bin/env python3
"""Render a Physion++ probe prediction/GT 2D-box overlay for a seeded video."""
from __future__ import annotations
import argparse,hashlib,json,pickle,random
from pathlib import Path
import cv2,numpy as np,torch
from .model import PhysionDecoder,match_objects
from .prepare_targets import targets,decode_rle

def choose(cache_root,split,seed):
 files=sorted((cache_root/split).glob('*.pt'))
 if not files:raise FileNotFoundError(f'no latent caches under {cache_root/split}')
 i=random.Random(seed).randrange(len(files));record=torch.load(files[i],map_location='cpu',weights_only=True);return Path(record['path']),record,i,len(files)
def cache_name(path):return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:16]+'.pt'
def box(frame,rgb):
 m=np.abs(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.int16)-np.asarray(rgb).reshape(1,1,3)).max(2)<=8;ys,xs=np.where(m);return None if not len(xs) else (xs.min(),ys.min(),xs.max(),ys.max())
def draw(im,b,c,label):
 if b is not None: x0,y0,x1,y1=map(int,b);cv2.rectangle(im,(x0,y0),(x1,y1),c,2,cv2.LINE_AA);cv2.putText(im,label,(x0,max(14,y0-3)),cv2.FONT_HERSHEY_SIMPLEX,.38,c,1,cv2.LINE_AA)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,default=Path('/data/ABDUCTIVE-WORLD/physion_v2/extracted'));p.add_argument('--split',default='readout_data_v1');p.add_argument('--seed',type=int,default=239);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();video,record,idx,count=choose(a.cache_root,a.split,a.seed);meta=pickle.loads(video.with_name(video.name[:-8]+'.pkl').read_bytes());target=targets(video);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=PhysionDecoder(**payload.get('model_config',{})).to(device);model.load_state_dict(payload['model'],strict=True);model.eval();batch={k:torch.tensor(bool(v)).unsqueeze(0) if np.asarray(v).ndim==0 else torch.tensor(v).unsqueeze(0) for k,v in target.items() if k in ('object_present','state_2d','state_valid','pair_distance_2d','pair_valid','contact','contact_valid','first_contact_class','ocp_label','ocp_valid')};
 batch={k:v.to(device) for k,v in batch.items()}
 with torch.no_grad():out=model(record['context_tokens'][None].to(device),record['future_tokens'][None].to(device));assign=match_objects(out,batch);pred=out['trajectory_2d'][0].cpu();assign=assign[0].cpu();
 current_indices=np.asarray(record['current_indices']).astype(np.int64).tolist();future_indices=np.asarray(record['future_indices']).astype(np.int64).tolist();all_indices=current_indices+future_indices;cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS) or 30.;w,h=int(cap.get(3)),int(cap.get(4));a.output_dir.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(a.output_dir/'prediction_overlay.mp4v.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));objects=meta['static'].get('video_object_segmentation_colors',[])
 for output_step,raw in enumerate(all_indices):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(raw));ok,frame=cap.read()
  if not ok:raise RuntimeError(f'failed reading frame {raw}')
  image=frame.copy();phase='current' if output_step<len(current_indices) else 'future';step=output_step-len(current_indices)
  if phase=='future':
   entries={int(e['idx']):e for e in json.loads(video.with_name(video.name[:-8]+'_id.json').read_text()).get(f'{int(raw):04d}',[])}
   for i in range(int(batch['object_present'][0].sum())):
    e=entries.get(i);gt=None
    if e is not None:
     m=decode_rle(e);ys,xs=np.where(m);gt=None if not len(xs) else (xs.min(),ys.min(),xs.max(),ys.max())
    draw(image,gt,(0,220,0),f'GT {i}');slot=int((assign==i).nonzero()[0]) if (assign==i).any() else -1
    if slot>=0:
     s=pred[slot,step];cx,cy,bw,bh=s[:4];draw(image,((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h),(220,0,220),f'P {slot}')
  cv2.putText(image,f'{phase} frame={int(raw)} seed={a.seed}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA);writer.write(image)
 cap.release();writer.release();import imageio_ffmpeg,subprocess;final=a.output_dir/'prediction_overlay.mp4';subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(a.output_dir/'prediction_overlay.mp4v.mp4'),'-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(final)],check=True);(a.output_dir/'prediction_overlay.mp4v.mp4').unlink();(a.output_dir/'manifest.json').write_text(json.dumps({'seed':a.seed,'video':str(video),'video_selection_index':idx,'video_candidate_count':count,'checkpoint':str(a.checkpoint.resolve()),'current_indices':current_indices,'future_indices':future_indices,'output':str(final)},indent=2)+'\n');print(json.dumps({'video':str(video),'output':str(final),'current_frames':len(current_indices),'future_frames':len(future_indices)},indent=2))
if __name__=='__main__':main()
