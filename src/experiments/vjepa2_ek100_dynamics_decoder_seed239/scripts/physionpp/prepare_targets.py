#!/usr/bin/env python3
"""Build Physion++ 2D object, pair, collision, and OCP targets."""
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
import numpy as np
import torch
try:
    from .model import FUTURE_STEPS, MAX_OBJECTS
except ImportError:
    from .model import FUTURE_STEPS, MAX_OBJECTS

def decode_rle(rle):
    h,w=map(int,rle['size']); s=rle['counts']; counts=[]; pos=0
    while pos<len(s):
        value=shift=0
        while True:
            code=ord(s[pos])-48;pos+=1;value|=(code&31)<<(5*shift)
            if not code&32:
                if code&16:value|=-1<<(5*(shift+1))
                break
            shift+=1
        counts.append(value+counts[-2] if len(counts)>2 else value)
    flat=np.zeros(h*w,np.uint8);off=0
    for i,run in enumerate(counts):
        if run<0 or off+run>flat.size:raise ValueError('invalid RLE')
        if i%2:flat[off:off+run]=1
        off+=run
    if off!=flat.size:raise ValueError('RLE size mismatch')
    return flat.reshape((h,w),order='F').astype(bool)

def targets(video,clip_frames=16,current_step=2,future_step=4,gap=0):
    pkl=video.with_name(video.name[:-8]+'.pkl'); ids=video.with_name(video.name[:-8]+'_id.json')
    with pkl.open('rb') as f:meta=pickle.load(f)
    annotations=json.loads(ids.read_text())
    static=meta['static']; anchor=int(static['start_frame_for_prediction']); future=anchor+gap+np.arange(clip_frames)*future_step; object_ids=np.asarray(static['object_ids']).astype(int); n=len(object_ids); normal_seg_count=len(static.get('video_object_segmentation_colors', object_ids))
    if n>MAX_OBJECTS:raise ValueError(f'{video} has {n} objects')
    present=np.zeros(MAX_OBJECTS,bool);present[:n]=True; state=np.zeros((MAX_OBJECTS,FUTURE_STEPS,7),np.float32);valid=np.zeros((MAX_OBJECTS,FUTURE_STEPS),bool)
    ignored_obi_entries=0
    for step,frame in enumerate(future):
        for item in annotations.get(f'{int(frame):04d}',[]):
            idx=int(item['idx'])
            if idx<0 or idx>=normal_seg_count:
                ignored_obi_entries+=1
                continue
            mask=decode_rle(item);ys,xs=np.where(mask)
            if not len(xs):continue
            h,w=mask.shape;x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max(); state[idx,step,:5]=[(xs.mean()+.5)/w,(ys.mean()+.5)/h,(x1-x0+1)/w,(y1-y0+1)/h,mask.mean()];valid[idx,step]=True
    for idx in range(n):
        for step in np.flatnonzero(valid[idx]):
            if step and valid[idx,step-1]:state[idx,step,5:]=state[idx,step,:2]-state[idx,step-1,:2]
    distance=np.zeros((MAX_OBJECTS,MAX_OBJECTS,FUTURE_STEPS),np.float32);pair_valid=np.zeros_like(distance,dtype=bool);contact=np.zeros_like(distance);contact_valid=np.zeros_like(distance,dtype=bool);first=np.full((MAX_OBJECTS,MAX_OBJECTS),-1,np.int64);lookup={oid:i for i,oid in enumerate(object_ids)}
    for i in range(n):
        for j in range(i+1,n):
            pv=valid[i]&valid[j];pair_valid[i,j]=pair_valid[j,i]=pv;distance[i,j,pv]=distance[j,i,pv]=np.linalg.norm(state[i,pv,:2]-state[j,pv,:2],axis=-1);contact_valid[i,j]=contact_valid[j,i]=True; hit=[]
            for step,frame in enumerate(future):
                fmeta=meta['frames'].get(f'{int(frame):04d}',{}); pairs=np.asarray(fmeta.get('collisions',{}).get('object_ids',[]))
                if pairs.ndim==2 and pairs.shape[1]==2 and any({lookup.get(int(a),-1),lookup.get(int(b),-1)}=={i,j} for a,b in pairs):contact[i,j,step]=contact[j,i,step]=1;hit.append(step)
            first[i,j]=first[j,i]=hit[0] if hit else FUTURE_STEPS
    label=static.get('does_target_contact_zone');return {'object_present':present,'state_2d':state,'state_valid':valid,'pair_distance_2d':distance,'pair_valid':pair_valid,'contact':contact,'contact_valid':contact_valid,'first_contact_class':first,'ocp_label':np.float32(bool(label)),'ocp_valid':np.bool_(label is not None),'anchor':anchor,'future_indices':future.astype(np.int64),'ignored_obi_entries':np.int64(ignored_obi_entries)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--split',choices=('data_v1','readout_data_v1','testdata_v1'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-videos',type=int);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--skip-missing-annotations',action='store_true');a=p.parse_args();videos=sorted((a.dataset_root/a.split).glob('**/*_img.mp4'));videos=videos[:a.max_videos] if a.max_videos else videos;videos=videos[a.rank::a.world_size]; rows=[]; skipped=[]
    for video in videos:
        ids=video.with_name(video.name[:-8]+'_id.json')
        if not ids.is_file():
            if not a.skip_missing_annotations:
                raise FileNotFoundError(f'missing segmentation annotation for {video}: expected {ids}')
            skipped.append(str(video.resolve()))
            continue
        row=targets(video);row['path']=str(video.resolve());rows.append(row)
    if not rows:
        raise RuntimeError(f'no usable videos found in split {a.split}; skipped={len(skipped)}')
    keys=('object_present','state_2d','state_valid','pair_distance_2d','pair_valid','contact','contact_valid','first_contact_class','ocp_label','ocp_valid','anchor','future_indices');payload={'protocol':'physionpp_segmentation_2d_targets_v2','split':a.split,'path':[r['path'] for r in rows],'skipped_missing_annotations':skipped,'ignored_obi_entries':int(sum(int(r['ignored_obi_entries']) for r in rows)),**{k:torch.from_numpy(np.stack([r[k] for r in rows])) for k in keys}};a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,a.output);print(json.dumps({'output':str(a.output),'samples':len(rows),'skipped_missing_annotations':skipped,'ignored_obi_entries':payload['ignored_obi_entries']},indent=2))
if __name__=='__main__':main()
