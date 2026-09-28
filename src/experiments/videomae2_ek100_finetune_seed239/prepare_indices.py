from __future__ import annotations
import argparse,json
from pathlib import Path
from .dataset import split_indices
from src.core.run_context import apply_cli_defaults, task_context

def main():
 ctx=task_context();p=argparse.ArgumentParser(); p.add_argument('--annotations',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=239); apply_cli_defaults(p,ctx);a=p.parse_args(); tr,va=split_indices(a.annotations,seed=a.seed); Path(a.output).write_text(json.dumps({'train':tr,'probe_validation':va},indent=2)+'\n')
if __name__=='__main__': main()
