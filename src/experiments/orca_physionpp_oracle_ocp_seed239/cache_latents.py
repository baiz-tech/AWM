#!/usr/bin/env python3
"""Cache Orca context and ground-truth future video latents for Physion++."""
from __future__ import annotations
import argparse, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np, torch
from decord import VideoReader, cpu
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.core.orca_model import load_orca, extract_video_feature
from src.core.run_context import apply_cli_defaults, task_context

def cname(path): return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:16] + ".pt"
def label(path):
    with open(str(path).replace("_img.mp4", ".pkl"), "rb") as f: m = pickle.load(f)
    return int(bool(m.get("static", {}).get("does_target_contact_zone", False)))
def main():
    ctx=task_context();p=argparse.ArgumentParser(); p.add_argument("--physion-root",type=Path,required=True); p.add_argument("--split",required=True,choices=["data_v1","readout_data_v1","testdata_v1"]); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--checkpoint-dir",type=Path,default=Path("/data/shared_model/Orca-4B")); p.add_argument("--clip-frames",type=int,default=16); p.add_argument("--current-step",type=int,default=2); p.add_argument("--future-step",type=int,default=4); p.add_argument("--prompt",default="Represent the observed physical state and dynamics."); p.add_argument("--max-samples",type=int); p.add_argument("--resume",action="store_true"); p.add_argument("--rank",type=int,default=0); p.add_argument("--world-size",type=int,default=1); apply_cli_defaults(p,ctx);a=p.parse_args()
    if a.rank < 0 or a.rank >= a.world_size: raise ValueError("rank must satisfy 0 <= rank < world_size")
    out=a.output_dir/a.split; out.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu");
    # Each launcher worker receives one physical GPU through CUDA_VISIBLE_DEVICES;
    # inside that process the selected device is therefore always ordinal 0.
    if device.type == "cuda": torch.cuda.set_device(0)
    model,proc=load_orca(a.checkpoint_dir,device); paths=sorted((a.physion_root/a.split).glob("**/*_img.mp4")); paths=paths[:a.max_samples] if a.max_samples else paths; paths=paths[a.rank::a.world_size]; records=[]
    for i,path in enumerate(paths):
        target=out/cname(path)
        if a.resume and target.is_file(): continue
        with open(str(path).replace("_img.mp4",".pkl"),"rb") as f: meta=pickle.load(f)
        anchor=int(meta["static"]["start_frame_for_prediction"]); ci=(anchor-a.clip_frames*a.current_step+np.arange(a.clip_frames)*a.current_step).astype(int); fi=(anchor+np.arange(a.clip_frames)*a.future_step).astype(int)
        vr=VideoReader(str(path),num_threads=1,ctx=cpu(0)); n=len(vr)
        if ci[0]<0 or fi[-1]>=n: continue
        cf=vr.get_batch(ci).asnumpy(); ff=vr.get_batch(fi).asnumpy(); c,ct=extract_video_feature(model,proc,cf,a.prompt,device); f,ft=extract_video_feature(model,proc,ff,a.prompt,device)
        torch.save({"protocol":"orca_physionpp_groundtruth_future_latent_v1","path":str(path.resolve()),"anchor_frame":anchor,"current_indices":ci.tolist(),"future_indices":fi.tolist(),"context_latent":c.half(),"future_latent":f.half(),"context_token_count":ct,"future_token_count":ft,"ocp_label":label(path),"ocp_valid":True},target); records.append(str(path));
        if i==0 or (i+1)%10==0: print(f"rank={a.rank} cached {i+1}/{len(paths)}",flush=True)
    (out/f"manifest.rank{a.rank}.json").write_text(json.dumps({"protocol":"orca_physionpp_groundtruth_future_latent_v1","split":a.split,"rank":a.rank,"world_size":a.world_size,"samples":len(records),"paths":records},indent=2)+"\n")
if __name__ == "__main__": main()
