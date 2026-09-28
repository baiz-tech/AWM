import argparse,json
from pathlib import Path
import torch
KEYS=('state_2d','state_valid','object_present','ocp_label','ocp_valid','anchor','future_indices','target_id','target_seg_idx')
def main():
 p=argparse.ArgumentParser();p.add_argument('--shard-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',required=True);a=p.parse_args();parts=[torch.load(x,map_location='cpu',weights_only=False) for x in sorted(a.shard_dir.glob('rank_*.pt'))];rows=[]
 for part in parts:
  for i,path in enumerate(part['path']):rows.append((path,part['target_source'][i],{k:part[k][i] for k in KEYS}))
 rows.sort(key=lambda x:x[0]);payload={'protocol':'physionpp_target_object_only_segmentation_v1','split':a.split,'path':[x[0] for x in rows],'target_source':[x[1] for x in rows],**{k:torch.stack([x[2][k] for x in rows]) for k in KEYS}};a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,a.output);print(json.dumps({'output':str(a.output),'samples':len(rows),'obi_targets':sum(x[1]=='obi_object_ids' for x in rows)},indent=2))
if __name__=='__main__':main()
