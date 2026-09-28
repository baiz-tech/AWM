#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
import numpy as np, torch
from .prepare_targets import decode_rle
STEPS=16
def build(video):
    stem=video.name[:-8]
    with video.with_name(stem+'.pkl').open('rb') as f: meta=pickle.load(f)
    ann=json.loads(video.with_name(stem+'_id.json').read_text()); s=meta['static']; object_ids=[int(x) for x in s.get('object_ids',[])]; obi_ids=[int(x) for x in s.get('obi_object_ids',[])]; target_id=int(s['target_id'])
    normal_count=len(s.get('video_object_segmentation_colors',object_ids)); source='object_ids' if target_id in object_ids else 'obi_object_ids' if target_id in obi_ids else None
    if source is None: raise ValueError(f'{video}: target_id={target_id} is not in object_ids or obi_object_ids')
    local_index=(object_ids if source=='object_ids' else obi_ids).index(target_id); seg_idx=local_index if source=='object_ids' else normal_count+local_index
    anchor=int(s['start_frame_for_prediction']); future=anchor+np.arange(STEPS)*4; state=np.zeros((STEPS,7),np.float32); valid=np.zeros(STEPS,bool)
    for t,frame in enumerate(future):
        item=next((x for x in ann.get(f'{int(frame):04d}',[]) if int(x['idx'])==seg_idx),None)
        if item is None: continue
        mask=decode_rle(item); ys,xs=np.where(mask)
        if len(xs):
            h,w=mask.shape; state[t,:5]=[(xs.mean()+.5)/w,(ys.mean()+.5)/h,(xs.max()-xs.min()+1)/w,(ys.max()-ys.min()+1)/h,mask.mean()];valid[t]=True
    for t in range(1,STEPS):
        if valid[t] and valid[t-1]: state[t,5:]=state[t,:2]-state[t-1,:2]
    return {'state_2d':state,'state_valid':valid,'object_present':np.bool_(True),'ocp_label':np.float32(bool(s.get('does_target_contact_zone'))),'ocp_valid':np.bool_(s.get('does_target_contact_zone') is not None),'anchor':np.int64(anchor),'future_indices':future.astype(np.int64),'target_id':np.int64(target_id),'target_seg_idx':np.int64(seg_idx),'target_source':source,'path':str(video.resolve())}
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--split',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--skip-missing-annotations',action='store_true');a=p.parse_args();videos=sorted((a.dataset_root/a.split).glob('**/*_img.mp4'))[a.rank::a.world_size];rows=[];skipped=[]
    for v in videos:
        ids=v.with_name(v.name[:-8]+'_id.json')
        if not ids.is_file():
            if not a.skip_missing_annotations: raise FileNotFoundError(ids)
            skipped.append(str(v.resolve()));continue
        rows.append(build(v))
    if not rows: raise RuntimeError(f'no target-only samples for {a.split}')
    keys=('state_2d','state_valid','object_present','ocp_label','ocp_valid','anchor','future_indices','target_id','target_seg_idx');payload={'protocol':'physionpp_target_object_only_segmentation_v1','split':a.split,'path':[r['path'] for r in rows],'target_source':[r['target_source'] for r in rows],'skipped_missing_annotations':skipped,**{k:torch.from_numpy(np.stack([r[k] for r in rows])) for k in keys}};a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,a.output);print(json.dumps({'output':str(a.output),'samples':len(rows),'obi_targets':sum(x=='obi_object_ids' for x in payload['target_source'])},indent=2))
if __name__=='__main__':main()
