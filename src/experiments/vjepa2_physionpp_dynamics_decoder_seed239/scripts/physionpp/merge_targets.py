#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
KEYS=('object_present','state_2d','state_valid','object_type','color_rgb','is_target','pair_distance_2d','pair_valid','contact','contact_valid','first_contact_class','ocp_label','ocp_valid','anchor','future_indices')
def main():
    p=argparse.ArgumentParser();p.add_argument('--shard-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',required=True);a=p.parse_args();shards=sorted(a.shard_dir.glob('rank_*.pt'))
    if not shards: raise FileNotFoundError(f'no target shards under {a.shard_dir}')
    parts=[torch.load(x,map_location='cpu',weights_only=False) for x in shards];order=sorted(range(len(parts)),key=lambda i: parts[i]['path'][0] if parts[i]['path'] else '');paths=[p for i in order for p in parts[i]['path']];payload={'protocol':'physionpp_segmentation_2d_targets_v3_metadata','split':a.split,'object_type_vocab':parts[order[0]].get('object_type_vocab',['unknown']),'path':paths,'skipped_missing_annotations':[p for i in order for p in parts[i].get('skipped_missing_annotations',[])],'ignored_obi_entries':sum(int(parts[i].get('ignored_obi_entries',0)) for i in order)}
    for k in KEYS: payload[k]=torch.cat([parts[i][k] for i in order],dim=0)
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,a.output);print(json.dumps({'output':str(a.output),'samples':len(paths),'ignored_obi_entries':payload['ignored_obi_entries'],'skipped_missing_annotations':len(payload['skipped_missing_annotations'])},indent=2))
if __name__=='__main__': main()
