#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from src.core.run_context import apply_cli_defaults, task_context

def load(root: Path, split: str):
    rows = [torch.load(p, map_location='cpu', weights_only=True) for p in sorted((root / split).glob('*.pt'))]
    if not rows: raise FileNotFoundError(f'no cached samples in {root / split}')
    x = torch.stack([torch.cat([r['context_latent'].float(), r['future_latent'].float()]) for r in rows])
    y = torch.tensor([r['ocp_label'] for r in rows], dtype=torch.float32)
    valid = torch.tensor([r['ocp_valid'] for r in rows], dtype=torch.bool)
    return x[valid], y[valid]

def metrics(prob, y, threshold):
    pred, truth = prob.ge(threshold), y.bool(); tp=int((pred&truth).sum()); tn=int((~pred&~truth).sum()); fp=int((pred&~truth).sum()); fn=int((~pred&truth).sum()); pos=tp+fn; neg=tn+fp
    precision=tp/max(tp+fp,1); recall=tp/max(pos,1); auroc=None
    if pos and neg:
        order=torch.argsort(prob); ranks=torch.empty(len(prob),dtype=torch.float64); s=prob[order]; start=0
        while start<len(prob):
            end=start+1
            while end<len(prob) and s[end]==s[start]: end+=1
            ranks[order[start:end]]=0.5*(start+1+end); start=end
        auroc=float((ranks[truth].sum()-pos*(pos+1)/2)/(pos*neg))
    return {'samples':len(y),'accuracy':(tp+tn)/max(len(y),1),'balanced_accuracy':.5*(recall+tn/max(neg,1)),'f1':2*precision*recall/max(precision+recall,1e-12),'auroc':auroc,'positive_recall':recall,'negative_recall':tn/max(neg,1),'tp':tp,'tn':tn,'fp':fp,'fn':fn}

def main():
    ctx=task_context();p=argparse.ArgumentParser(); p.add_argument('--cache-root',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--epochs',type=int,default=300); p.add_argument('--lr',type=float,default=1e-2); p.add_argument('--weight-decay',type=float,default=1e-3); p.add_argument('--seed',type=int,default=239); apply_cli_defaults(p,ctx);a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    train_x,train_y=load(a.cache_root,'data_v1'); val_x,val_y=load(a.cache_root,'readout_data_v1'); test_x,test_y=load(a.cache_root,'testdata_v1')
    mean,std=train_x.mean(0,keepdim=True),train_x.std(0,keepdim=True).clamp_min(1e-6); net=torch.nn.Linear(train_x.size(1),1); opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=a.weight_decay); pw=(len(train_y)-train_y.sum()).clamp_min(1)/train_y.sum().clamp_min(1); history=[]; best=float('inf'); best_state=None
    for epoch in range(1,a.epochs+1):
        net.train(); logits=net((train_x-mean)/std).squeeze(1); loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,train_y,pos_weight=pw); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); net.eval()
        with torch.no_grad(): val_logits=net((val_x-mean)/std).squeeze(1); val_loss=torch.nn.functional.binary_cross_entropy_with_logits(val_logits,val_y)
        row={'epoch':epoch,'train_split':'data_v1','validation_split':'readout_data_v1','train_samples':len(train_y),'validation_samples':len(val_y),'train_loss':float(loss),'validation_loss':float(val_loss)}; history.append(row); print(json.dumps(row),flush=True)
        if float(val_loss)<best: best=float(val_loss); best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad(): val_prob=torch.sigmoid(net((val_x-mean)/std).squeeze(1)); test_prob=torch.sigmoid(net((test_x-mean)/std).squeeze(1))
    thresholds=torch.unique(torch.cat([torch.tensor([.5]),val_prob])).sort().values; threshold=max(((metrics(val_prob,val_y,float(t))['balanced_accuracy'],-abs(float(t)-.5),float(t)) for t in thresholds))[2]; result=metrics(test_prob,test_y,threshold); result.update({'protocol':'orca_physionpp_groundtruth_future_ocp_v2','feature_definition':'concat(mean(context_video_tokens), mean(future_video_tokens))','future_latent_source':'ground_truth_future_video','train_split':'data_v1','threshold_split':'readout_data_v1','test_split':'testdata_v1','train_samples':len(train_y),'threshold_validation_samples':len(val_y),'test_samples':len(test_y),'threshold':threshold,'decoder_used':False,'shallow_probe_used':False,'orca_nfp_head_used':False}); a.output_dir.mkdir(parents=True,exist_ok=True); torch.save({'model':net.state_dict(),'mean':mean,'std':std,'threshold':threshold,'protocol':result['protocol']},a.output_dir/'readout.pt'); (a.output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n'); (a.output_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'final_test':result},indent=2),flush=True)
if __name__=='__main__': main()
