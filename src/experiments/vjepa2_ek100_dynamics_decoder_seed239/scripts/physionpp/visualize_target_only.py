#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import cv2,numpy as np,torch
from .model_target_only import TargetOnlyDecoder
from .prepare_targets import decode_rle
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--targets',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--split',default='readout_data_v1');p.add_argument('--seed',type=int,default=241);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();t=torch.load(a.targets,map_location='cpu',weights_only=False);i=random.Random(a.seed).randrange(len(t['path']));video=Path(t['path'][i]);import hashlib;c=a.cache_root/a.split/(hashlib.sha1(str(video.resolve()).encode()).hexdigest()[:16]+'.pt');r=torch.load(c,map_location='cpu',weights_only=True);pay=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=TargetOnlyDecoder(**pay.get('model_config',{})).cuda();m.load_state_dict(pay['model']);m.eval()
 with torch.no_grad():pred=m(r['context_tokens'][None].cuda(),r['future_tokens'][None].cuda())['trajectory_2d'][0].cpu()
 ann=json.loads(video.with_name(video.name[:-8]+'_id.json').read_text());cur=np.asarray(r['current_indices']).tolist();fut=np.asarray(r['future_indices']).tolist();current_step=int(np.median(np.diff(cur)));future_step=int(np.median(np.diff(fut)));future_repeats=max(1,round(future_step/current_step));cap=cv2.VideoCapture(str(video));w,h=int(cap.get(3)),int(cap.get(4));fps=cap.get(cv2.CAP_PROP_FPS) or 30;a.output_dir.mkdir(parents=True,exist_ok=True);tmp=a.output_dir/'tmp.mp4';wr=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));seg=int(t['target_seg_idx'][i])
 for k,fr in enumerate(cur+fut):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(fr));ok,im=cap.read();assert ok;phase='current' if k<16 else 'future'
  if phase=='future':
   st=k-16;item=next((x for x in ann.get(f'{int(fr):04d}',[]) if int(x['idx'])==seg),None)
   if item is not None:
    ys,xs=np.where(decode_rle(item));
    if len(xs):cv2.rectangle(im,(xs.min(),ys.min()),(xs.max(),ys.max()),(0,220,0),2)
   cx,cy,bw,bh=pred[st,:4];bw=max(0.,float(bw));bh=max(0.,float(bh));cv2.rectangle(im,(int((cx-bw/2)*w),int((cy-bh/2)*h)),(int((cx+bw/2)*w),int((cy+bh/2)*h)),(220,0,220),2)
  cv2.putText(im,f'{phase} frame={fr}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
  for _ in range(future_repeats if phase=='future' else 1):wr.write(im)
 cap.release();wr.release();import imageio_ffmpeg,subprocess;out=a.output_dir/'prediction_overlay.mp4';subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(tmp),'-c:v','libx264','-pix_fmt','yuv420p',str(out)],check=True);tmp.unlink();(a.output_dir/'manifest.json').write_text(json.dumps({'video':str(video),'target_id':int(t['target_id'][i]),'target_source':t['target_source'][i],'target_seg_idx':seg,'current_step':current_step,'future_step':future_step,'future_frame_repeats':future_repeats,'encoded_frames':len(cur)+len(fut)*future_repeats,'output':str(out)},indent=2)+'\n')
if __name__=='__main__':main()
