#!/usr/bin/env python3
"""Batch video interventions over a Physion++ split."""
from __future__ import annotations
import argparse, json, random, os
from pathlib import Path
import numpy as np, torch, yaml
from decord import VideoReader, cpu
from external.vjepa2.app.vjepa.transforms import make_transforms
from src.experiments.vjepa2_clevrer_naive_world_model_seed239.model import build_model
from .evaluate_video_intervention import _edit_frames, _expected_targets, _instance_masks, _metadata, _metrics, _object_masks, _paste_object, _read_frames
from .model import PhysionDecoder, match_objects
from .prepare_targets import targets

MODES = ("base", "background_replace", "horizontal_flip", "vertical_flip", "add_object")

def _vocab(video, checkpoint_vocab):
    if checkpoint_vocab: return list(checkpoint_vocab)
    names={"unknown"}; names.update(str(x.decode() if isinstance(x,bytes) else x) for x in _metadata(video)["static"].get("model_names",[])); return sorted(names)

def _run_one(video, donor, args, predictor, probe, transform, vocab, device):
    meta=_metadata(video); data=args.data_cfg; anchor=int(meta["static"]["start_frame_for_prediction"]); sc=int(data.get("current_frame_step",2)); sf=int(data.get("future_frame_step",4)); gap=int(data.get("clip_gap",0)); ci=anchor-16*sc+np.arange(16)*sc; fi=anchor+gap+np.arange(16)*sf; indices=np.concatenate((ci,fi)); frames=_read_frames(video,indices); fg=_instance_masks(video,indices)
    dmeta=_metadata(donor); dnames=[x.decode() if isinstance(x,bytes) else str(x) for x in dmeta["static"].get("model_names",[])]; dcount=len(dmeta["static"].get("video_object_segmentation_colors",dnames)); candidates=list(range(min(len(dnames),dcount))); dobj=random.Random(args.seed + anchor + 17).choice(candidates) if candidates else 0; da=int(dmeta["static"]["start_frame_for_prediction"]); dci=da-16*sc+np.arange(16)*sc; dfi=da+gap+np.arange(16)*sf; dindices=np.concatenate((dci,dfi)); donor_frames=_read_frames(donor,dindices); donor_mask=_object_masks(donor,dindices,dobj); added_frames=_paste_object(frames,donor_frames,donor_mask); vocab_map={name:i for i,name in enumerate(vocab)}; donor_row=targets(donor,type_to_id=vocab_map); dtype=dnames[dobj] if dobj<len(dnames) else "unknown"; added={"object_type":vocab_map.get(dtype,vocab_map.get("unknown",0)),"color_rgb":donor_row["color_rgb"][dobj],"state_2d":donor_row["state_2d"][dobj],"state_valid":donor_row["state_valid"][dobj]}; target_row=targets(video,type_to_id=vocab_map); rows={}
    for mode in MODES:
        edited=added_frames if mode=="add_object" else _edit_frames(frames,fg,mode); ten=transform(edited); c,f=ten[:,:16].unsqueeze(0).to(device),ten[:,16:].unsqueeze(0).to(device)
        with torch.inference_mode(): ctx=predictor.encode_current(c); fut=predictor.predict_next(ctx); out=probe(ctx.reshape(1,8,256,-1),fut.reshape(1,8,256,-1))
        clean={k:v[0].detach().cpu() for k,v in out.items() if torch.is_tensor(v)}; expected=_expected_targets(target_row,mode,added if mode=="add_object" else None); assignment=match_objects(out,{"object_present":torch.as_tensor(expected["object_present"],device=device).unsqueeze(0),"state_2d":torch.as_tensor(expected["state_2d"],device=device).unsqueeze(0),"state_valid":torch.as_tensor(expected["state_valid"],device=device).unsqueeze(0)})[0].cpu(); item=_metrics(clean,expected,assignment); item.update(predicted_present_count=int(clean["presence_logits"].sigmoid().ge(.5).sum()),expected_present_count=int(np.asarray(expected["object_present"]).sum()),assignment=assignment.tolist()); rows[mode]=item
    return {"video":str(video.resolve()),"donor_video":str(donor.resolve()),"donor_object_index":dobj,"donor_object_type":dtype,"anchor":anchor,"current_indices":ci.tolist(),"future_indices":fi.tolist(),"metrics":rows}

def _aggregate(records):
    keys=("presence_f1","object_type_accuracy","center_mae","geometry_mae","velocity_mae","pair_distance_mean","contact_probability_mean"); result={}
    for mode in MODES:
        vals=[r["metrics"][mode] for r in records]; result[mode]={k:float(np.mean([x[k] for x in vals if x[k] is not None])) if vals else None for k in keys}; result[mode]["mean_predicted_present_count"]=float(np.mean([x["predicted_present_count"] for x in vals])) if vals else None; result[mode]["mean_expected_present_count"]=float(np.mean([x["expected_present_count"] for x in vals])) if vals else None
    for mode in MODES[1:]: result[mode]["delta_vs_base"]={k:result[mode][k]-result["base"][k] for k in keys}
    return result

def _report(meta, agg):
    lines=["# Physion++ 全测试视频干预总结","",f"- Split：`{meta['split']}`",f"- Seed：`{meta['seed']}`",f"- 完成视频数：`{meta['videos']}`",f"- 失败视频数：`{meta['failed']}`","","## 汇总指标","","| 版本 | Presence F1 | 类别准确率 | Center MAE | Geometry MAE | Velocity MAE | 平均识别物体数 | 平均期望物体数 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        x=agg[mode]; lines.append(f"| `{mode}` | {x['presence_f1']:.4f} | {x['object_type_accuracy']:.4f} | {x['center_mae']:.4f} | {x['geometry_mae']:.4f} | {x['velocity_mae']:.4f} | {x['mean_predicted_present_count']:.3f} | {x['mean_expected_present_count']:.3f} |")
    lines += ["","## 相对 base 的变化","","| 版本 | Presence F1 | 类别准确率 | Center MAE | Geometry MAE | Velocity MAE |","|---|---:|---:|---:|---:|---:|"]
    for mode in MODES[1:]:
        d=agg[mode]["delta_vs_base"]; lines.append(f"| `{mode}` | {d['presence_f1']:+.4f} | {d['object_type_accuracy']:+.4f} | {d['center_mae']:+.4f} | {d['geometry_mae']:+.4f} | {d['velocity_mae']:+.4f} |")
    lines += ["","## 解读","","- `background_replace` 检查背景不变性。","- 翻转版本需结合坐标等变性分析。","- `add_object` 重点比较识别物体数是否增加，以及 donor 物体是否被识别。","- 汇总为逐视频 macro-average；完整逐视频结果见 `per_video.jsonl`。",""]; return "\n".join(lines)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset-root",type=Path,required=True); p.add_argument("--split",default="testdata_v1"); p.add_argument("--seed",type=int,default=239); p.add_argument("--max-videos",type=int); p.add_argument("--resume",action="store_true"); p.add_argument("--probe-checkpoint",type=Path,required=True); p.add_argument("--predictor-checkpoint",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda:0")
    a=p.parse_args(); rank=int(os.environ.get("RANK",0)); world=int(os.environ.get("WORLD_SIZE",1)); local=int(os.environ.get("LOCAL_RANK",rank));
    if world > 1:
        if not torch.cuda.is_available(): raise RuntimeError("8-GPU evaluation requires CUDA")
        torch.cuda.set_device(local); torch.distributed.init_process_group("nccl")
    a.output_dir.mkdir(parents=True,exist_ok=True); videos=sorted((a.dataset_root/a.split).glob("**/*_img.mp4")); videos=videos[:a.max_videos] if a.max_videos else videos; total_requested=len(videos); videos=videos[rank::world]
    if not videos: raise RuntimeError(f"no videos found under {a.dataset_root/a.split}")
    cfg=yaml.safe_load(a.config.read_text())["experiment"]; a.data_cfg=cfg["data"]; device=torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu"); payload=torch.load(a.probe_checkpoint,map_location="cpu",weights_only=False); probe=PhysionDecoder(**payload.get("model_config",{})).to(device); probe.load_state_dict(payload["model"],strict=True); probe.eval(); vocab=list(payload.get("object_type_vocab",[])) or _vocab(videos[0] if videos else Path("."),None); vocab.extend(f"class_{i}" for i in range(len(vocab),int(probe.probe.object_type_head.out_features))); predictor,_,_=build_model(device,cfg["model"],a.data_cfg,cfg["meta"]["pretrain_checkpoint"],cfg["meta"].get("encoder_checkpoint_key","target_encoder")); pp=torch.load(a.predictor_checkpoint,map_location="cpu",weights_only=False); predictor.predictor.load_state_dict(pp["predictor"],strict=True); predictor.eval(); transform=make_transforms(False,(1.,1.),(1.,1.),0.,False,False,int(a.data_cfg.get("crop_size",256))); path=a.output_dir/f"per_video.rank{rank:03d}.jsonl"; existing={}
    if a.resume and path.is_file():
        for line in path.read_text().splitlines():
            if line.strip(): x=json.loads(line); existing[x["video"]]=x
    records=[]
    with path.open("a" if a.resume else "w",encoding="utf-8") as handle:
        for i,video in enumerate(videos):
            if str(video.resolve()) in existing: records.append(existing[str(video.resolve())]); continue
            pool=[x for x in videos if x.resolve()!=video.resolve()]; donor=random.Random(a.seed+i*1009+1).choice(pool or videos)
            try: row=_run_one(video,donor,a,predictor,probe,transform,vocab,device)
            except Exception as exc: row={"video":str(video.resolve()),"error":f"{type(exc).__name__}: {exc}"}
            handle.write(json.dumps(row)+"\n"); handle.flush(); records.append(row); print(json.dumps({"index":i,"videos":len(videos),"video":row["video"],"error":row.get("error")}),flush=True)
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        all_records=[]
        for rank_id in range(world):
            rank_path=a.output_dir/f"per_video.rank{rank_id:03d}.jsonl"
            if rank_path.is_file():
                all_records.extend(json.loads(line) for line in rank_path.read_text().splitlines() if line.strip())
        all_records.sort(key=lambda x: x["video"]); (a.output_dir/"per_video.jsonl").write_text("".join(json.dumps(x)+"\n" for x in all_records),encoding="utf-8")
        good=[x for x in all_records if "error" not in x]; meta={"protocol":"physionpp3_all_video_intervention_v1","split":a.split,"seed":a.seed,"videos":len(good),"failed":len(all_records)-len(good),"requested_videos":total_requested,"world_size":world,"probe_checkpoint":str(a.probe_checkpoint.resolve()),"predictor_checkpoint":str(a.predictor_checkpoint.resolve()),"config":str(a.config.resolve())}; agg=_aggregate(good); (a.output_dir/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n"); (a.output_dir/"aggregate.json").write_text(json.dumps(agg,indent=2)+"\n"); (a.output_dir/"summary.md").write_text(_report(meta,agg),encoding="utf-8"); print(json.dumps({"output_dir":str(a.output_dir.resolve()),"completed":len(good),"failed":len(all_records)-len(good),"aggregate":agg},indent=2))
    if world > 1:
        torch.distributed.destroy_process_group()

if __name__=="__main__": main()
