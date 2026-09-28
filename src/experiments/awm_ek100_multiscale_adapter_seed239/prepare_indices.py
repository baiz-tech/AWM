#!/usr/bin/env python3
"""Create deterministic video-disjoint EK100 train/probe-validation indices."""
import argparse, json
from pathlib import Path
import numpy as np
from src.data.ek100.dataset import read_rows, split_by_video
from src.core.run_context import apply_cli_defaults, task_context

def main():
    ctx=task_context();p=argparse.ArgumentParser(); p.add_argument('--annotations',required=True); p.add_argument('--output',required=True); p.add_argument('--fraction',type=float,default=.1); p.add_argument('--seed',type=int,default=239); apply_cli_defaults(p,ctx);a=p.parse_args()
    rows=read_rows(a.annotations); train,val=split_by_video(a.annotations,a.fraction,a.seed); Path(a.output).write_text(json.dumps({'train':train,'probe_validation':val,'final_validation':list(range(len(rows)))},indent=2)+'\n'); print(json.dumps({'train':len(train),'probe_validation':len(val),'final_validation':len(rows)}))
if __name__=='__main__': main()
