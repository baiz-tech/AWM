from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import yaml
from .model_variants import AttributionExtractor, AttributionHead, VARIANTS, load_components
from src.data.ek100.dataset import read_rows
from src.core.run_context import apply_cli_defaults, task_context

def main():
    ctx=task_context();p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--readout',required=True); p.add_argument('--adapter',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cpu'); apply_cli_defaults(p,ctx);a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); rows=read_rows(cfg['data']['train_annotations']); vv=sorted({int(r['verb_class']) for r in rows}); nv=sorted({int(r['noun_class']) for r in rows}); pairs=sorted({(vv.index(int(r['verb_class'])),nv.index(int(r['noun_class']))) for r in rows}); device=torch.device(a.device); comp=load_components(a.readout,a.adapter,device); result={'protocol':'ek100_fair_attribution_parameter_report_v1','variants':{}}
    for name in VARIANTS:
        ex=AttributionExtractor(name,comp.readout_world,comp.shared); head=AttributionHead(len(vv),len(nv),pairs); result['variants'][name]={'head_parameters':sum(x.numel() for x in head.parameters()),'extractor_parameters':sum(x.numel() for x in ex.parameters()),'trainable_head_parameters':sum(x.numel() for x in head.parameters() if x.requires_grad),'trainable_extractor_parameters':sum(x.numel() for x in ex.parameters() if x.requires_grad)}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
