#!/usr/bin/env python3
"""Context-only evidence ablations with frozen Predictor regeneration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.core.backbone import build_model
from src.experiments.awm_physionpp_fullpatch_probe_seed239.structured_probe import PhysionStructuredProbe
from .physion_input_ablation import mask_object_evidence, mask_random_evidence, summarize, temporal_static
from .physion_slot_time_probe import assignment
from src.core.run_context import apply_cli_defaults, task_context


def main():
    ctx = task_context()
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True); p.add_argument("--probe", required=True); p.add_argument("--predictor", required=True)
    p.add_argument("--predictor-config", required=True); p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int); p.add_argument("--batch-size", type=int, default=2); p.add_argument("--ratio", type=float, default=.5); p.add_argument("--device", default="cuda:0")
    apply_cli_defaults(p, ctx)
    a = p.parse_args(); device=torch.device(a.device); files=sorted(Path(a.cache).glob("sample_*.pt")); files=files[:a.max_samples] if a.max_samples else files
    cfg=torch.load(a.predictor,map_location="cpu",weights_only=False).get("config",{})
    import yaml
    full_cfg=yaml.safe_load(Path(a.predictor_config).read_text()); model,_,_=build_model(device,full_cfg["model"],full_cfg["data"],full_cfg["meta"]["pretrain_checkpoint"],full_cfg["meta"].get("encoder_checkpoint_key","target_encoder"))
    payload=torch.load(a.predictor,map_location="cpu",weights_only=False); model.predictor.load_state_dict(payload["predictor"],strict=True); model.eval()
    probe=PhysionStructuredProbe().to(device); probe_payload=torch.load(a.probe,map_location="cpu",weights_only=False); probe.load_state_dict(probe_payload.get("model",probe_payload),strict=True); probe.eval()
    all_results={name:[] for name in ("baseline_cached","baseline_regenerated","object_mask_regenerated","random_mask_regenerated","temporal_static_regenerated")}
    with torch.inference_mode():
        for start in range(0,len(files),a.batch_size):
            batch_rows=[torch.load(path,map_location="cpu",weights_only=False) for path in files[start:start+a.batch_size]]
            context=torch.stack([row["context"] for row in batch_rows]).to(device).float()
            cached_future=torch.stack([row["future"] for row in batch_rows]).to(device).float()
            base=probe(context,cached_future,return_features=True,return_attention=True); attention=base["object_memory_attention"]
            mask_context,_=mask_object_evidence(context,context.clone(),attention,a.ratio)
            random_context,_=mask_random_evidence(context,context.clone(),a.ratio)
            static_context,_=temporal_static(context,context.clone())
            variants={"baseline_cached":(context,cached_future),"baseline_regenerated":(context,model.predict_from_context(context.reshape(context.size(0),-1,context.size(-1)).float()).reshape_as(cached_future)),"object_mask_regenerated":(mask_context,model.predict_from_context(mask_context.reshape(mask_context.size(0),-1,mask_context.size(-1)).float()).reshape_as(cached_future)),"random_mask_regenerated":(random_context,model.predict_from_context(random_context.reshape(random_context.size(0),-1,random_context.size(-1)).float()).reshape_as(cached_future)),"temporal_static_regenerated":(static_context,model.predict_from_context(static_context.reshape(static_context.size(0),-1,static_context.size(-1)).float()).reshape_as(cached_future))}
            base_assignment=[assignment({"trajectory_prediction":base["trajectory"][i].cpu(),"future_object_state":row["future_object_state"],"future_object_valid":row["future_object_valid"]}) for i,row in enumerate(batch_rows)]
            for name,(cx,fy) in variants.items():
                output=probe(cx,fy,return_features=True)
                for i,row in enumerate(batch_rows): all_results[name].append({"output":{k:v[i].detach().cpu() for k,v in output.items() if torch.is_tensor(v)},"row":row,"assignment":base_assignment[i]})
    summary={}
    for name,items in all_results.items(): summary[name]=summarize([x["output"] for x in items],[x["row"] for x in items],[x["assignment"] for x in items])
    base=summary["baseline_cached"]; delta={name:{key:(None if summary[name][key] is None or base[key] is None else summary[name][key]-base[key]) for key in base} for name in summary if name!="baseline_cached"}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);(out.with_suffix(".json")).write_text(json.dumps({"protocol":"physion_predictor_regeneration_ablation_v1","samples":len(files),"batch_size":a.batch_size,"ratio":a.ratio,"summary":summary,"delta_vs_cached_baseline":delta},indent=2)+'\n');print(json.dumps({"summary":summary,"delta_vs_cached_baseline":delta},indent=2))

if __name__=="__main__":main()
