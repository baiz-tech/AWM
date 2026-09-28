#!/usr/bin/env python3
"""CLEVRER paired object-pair mask intervention with predictor regeneration."""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.experiments.awm_clevrer_fullpatch_probe_seed239.model import CLEVRERDecoder
from src.experiments.awm_clevrer_fullpatch_probe_seed239.structured_probe_evaluation import load_targets
from src.experiments.vjepa2_clevrer_naive_world_model_seed239.model import build_model
from src.core.run_context import apply_cli_defaults, task_context

PAIRS = list(itertools.combinations(range(6), 2))

def decode_rle(rle):
    h, w = map(int, rle['size']); enc=rle['counts']; counts=[]; pos=0
    while pos < len(enc):
        value=shift=0
        while True:
            code=ord(enc[pos])-48; pos+=1; value |= (code & 31) << (5*shift)
            if not code & 32:
                if code & 16: value |= -1 << (5*(shift+1))
                break
            shift += 1
        counts.append(value + counts[-2] if len(counts)>2 else value)
    flat=np.zeros(h*w,dtype=np.uint8); off=0
    for i,run in enumerate(counts):
        flat[off:off+run] = i % 2; off += run
    return flat.reshape((h,w), order='F').astype(bool)

def supports(annotation, slots, current_indices):
    by_key={(str(o['color']),str(o['material']),str(o['shape'])):i for i,o in enumerate(annotation['ground_truth']['objects'])}
    out=np.zeros((6,256),bool)
    for frame in current_indices:
        for obj in annotation['frames'][int(frame)]['objects']:
            slot=by_key.get((str(obj['color']),str(obj['material']),str(obj['shape'])))
            if slot is None or float(obj.get('score',1.0)) < .5: continue
            mask=decode_rle(obj['mask']); ten=torch.from_numpy(mask[None,None].astype(np.float32))
            ten=F.interpolate(ten,size=(256,256),mode='nearest')[0,0].numpy().reshape(16,16,16,16)
            out[slot] |= ten.any((1,3)).reshape(-1)
    return torch.from_numpy(out)

def choose_pairs(target, support, seed, area_tolerance=0.25):
    n=int(target['object_present'].sum()); valid=target['contact_valid'].bool(); contact=target['contact'].float()
    vals=[]
    for p in itertools.combinations(range(n),2):
        if not valid[p].any(): continue
        vals.append((p,bool(contact[p].any()),int((support[p[0]]|support[p[1]]).sum())))
    c=[x for x in vals if x[1] and x[2]]; nc=[x for x in vals if not x[1] and x[2]]
    if not c or not nc:return None
    cp=max(c,key=lambda x:x[2]); tolerance=max(1,int(round(cp[2]*area_tolerance)))
    matched=[x for x in nc if abs(x[2]-cp[2]) <= tolerance]
    if not matched: return None
    ncp=min(matched,key=lambda x:abs(x[2]-cp[2])); pool=[x for x in vals if x[0] not in {cp[0],ncp[0]} and abs(x[2]-cp[2]) <= tolerance] or matched
    rng=np.random.default_rng(seed); rp=pool[int(rng.integers(len(pool)))]
    return {'contact':cp[0],'noncontact':ncp[0],'random':rp[0],'areas':{'contact':cp[2],'noncontact':ncp[2],'random':rp[2]}}

def mask_pair(x,support,pair):
    ids=torch.where(support[pair[0]]|support[pair[1]])[0].to(x.device); y=x.clone()
    if len(ids): y[:,:,ids]=y.mean((1,2),keepdim=True)
    return y

def main():
    ctx=task_context();p=argparse.ArgumentParser(); p.add_argument('--cache-root',type=Path,required=True); p.add_argument('--targets',type=Path,required=True); p.add_argument('--probe',type=Path,required=True); p.add_argument('--world-checkpoint',type=Path,required=True); p.add_argument('--config',type=Path,required=True); p.add_argument('--dataset-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--max-scenes',type=int); p.add_argument('--device',default='cuda:0'); p.add_argument('--seeds',default='239,241,251'); p.add_argument('--area-tolerance',type=float,default=.25); apply_cli_defaults(p,ctx);a=p.parse_args()
    device=torch.device(a.device); target=load_targets(a.targets); limit=len(target['scene_id']) if a.max_scenes is None else min(len(target['scene_id']),a.max_scenes)
    probe_payload=torch.load(a.probe,map_location='cpu',weights_only=False); probe=CLEVRERDecoder(**probe_payload['model_config']).to(device); probe.load_state_dict(probe_payload['model'],strict=True); probe.eval()
    cfg=json.loads(a.config.read_text()) if a.config.suffix=='.json' else __import__('yaml').safe_load(a.config.read_text()); exp=cfg['tasks']['train']['experiment']; world,_,_=build_model(device,exp['model'],exp['data'],exp['meta']['pretrain_checkpoint'],exp['meta'].get('encoder_checkpoint_key','target_encoder')); world.predictor.load_state_dict(torch.load(a.world_checkpoint,map_location='cpu',weights_only=False)['predictor'],strict=True); world.eval()
    variants=('baseline','contact_pair','noncontact_pair','random_pair'); rows={k:[] for k in variants}; selections=[]; seeds=[int(x) for x in a.seeds.split(',') if x.strip()]
    with torch.inference_mode():
        for i in range(limit):
            sid=int(target['scene_id'][i]); ws=int(target['window_start'][i]); rec=torch.load(a.cache_root/'validation'/f'scene_{sid:05d}_window_{ws:03d}.pt',map_location='cpu',weights_only=True); ann=json.loads((a.dataset_root/'processed_proposals'/f'sim_{sid:05d}.json').read_text()); sup=supports(ann,target,rec['current_indices'].tolist()); cx=rec['context_tokens'].unsqueeze(0).float().to(device); fy=rec['future_tokens'].unsqueeze(0).float().to(device); base=probe(cx,fy); batch={k:v[i:i+1].to(device) for k,v in target.items() if torch.is_tensor(v) and k in ('object_present','color','material','shape','state','state_valid')}; assign=__import__('src.experiments.awm_clevrer_fullpatch_probe_seed239.scripts.clevrer.model',fromlist=['match_objects']).match_objects(base,batch)[0].cpu()
            for seed in seeds:
                pair=choose_pairs({k:v[i] for k,v in target.items() if torch.is_tensor(v)},sup,seed+i,a.area_tolerance)
                if pair is None: continue
                selections.append({'scene_id':sid,'window_start':ws,'seed':seed,**pair})
                inputs={'baseline':(cx,fy)}
                for name,key in (('contact_pair','contact'),('noncontact_pair','noncontact'),('random_pair','random')):
                    mc = mask_pair(cx, sup, pair[key]); mc_flat = mc.reshape(mc.size(0), -1, mc.size(-1)); mf = world.predict_next(mc_flat).reshape_as(fy); inputs[name]=(mc,mf)
                for name,(xx,ff) in inputs.items():
                    o=probe(xx,ff); item={'scene_id':sid,'window_start':ws,'seed':seed,'output':{k:v[0].cpu() for k,v in o.items() if torch.is_tensor(v)},'target':{k:v[i] for k,v in target.items() if torch.is_tensor(v)},'assignment':assign,'pair':pair}; rows[name].append(item)
    def metrics(items):
        dist=[]; contact=[]; labels=[]
        for it in items:
            o,t,m=it['output'],it['target'],it['assignment']; n=int(t['object_present'].sum()); inv={int(target_slot): slot for slot,target_slot in enumerate(m.tolist()) if 0 <= target_slot < n}
            for u,v in itertools.combinations(range(n),2):
                if u not in inv or v not in inv: continue
                pi=PAIRS.index(tuple(sorted((inv[u],inv[v]))))
                d=t['pair_distance'][u,v]; valid=t['pair_valid'][u,v].bool(); dist += [float(x) for x in (o['pair_distance_2d'][pi][valid]-d[valid]).abs()]
                cv=t['contact_valid'][u,v].bool(); labels += [float(x) for x in t['contact'][u,v][cv]]; contact += [float(x) for x in o['contact_gt_event'][pi][cv]]
        auc=None
        if labels and len(set(labels))>1:
            y=np.asarray(labels,bool); s=np.asarray(contact); order=np.argsort(np.argsort(s))+1; pos=y.sum(); auc=float((order[y].sum()-pos*(pos+1)/2)/(pos*(len(y)-pos)))
        return {'pair_distance_mae':float(np.mean(dist)) if dist else None,'contact_auroc':auc,'scenes':len(items)}
    summary={k:metrics(v) for k,v in rows.items()}; base=summary['baseline']; delta={k:{m:(None if summary[k][m] is None or base[m] is None else summary[k][m]-base[m]) for m in ('pair_distance_mae','contact_auroc')} for k in variants if k!='baseline'}; out=a.output; out.parent.mkdir(parents=True,exist_ok=True); out.with_suffix('.json').write_text(json.dumps({'protocol':'clevrer_true_instance_relation_pair_ablation_v2','input_samples':limit,'seeds':seeds,'area_tolerance':a.area_tolerance,'eligible_triplets':len(selections),'summary':summary,'delta_vs_baseline':delta,'pair_selection':selections},indent=2)); print(json.dumps({'summary':summary,'delta_vs_baseline':delta,'eligible_triplets':len(selections)},indent=2))
if __name__=='__main__': main()
