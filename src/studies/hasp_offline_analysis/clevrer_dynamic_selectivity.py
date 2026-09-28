"""Offline CLEVRER Dynamic selectivity analysis."""
from __future__ import annotations

import argparse, json, os, warnings, logging, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.experiments.awm_clevrer_fullpatch_probe_seed239.model import CLEVRERDecoder
from src.core.run_context import apply_cli_defaults, task_context

PAIR_COUNT, OBJECT_COUNT, STEPS = 15, 6, 16
LOGGER = logging.getLogger("clevrer_dynamic_selectivity")


def load_targets(path: str, limit: int | None = None) -> dict:
    z = torch.load(path, map_location="cpu", weights_only=False)
    n = len(z["scene_id"]); ids = torch.arange(n) if limit is None else torch.arange(min(n, limit))
    return {k: (v[ids] if torch.is_tensor(v) and v.ndim else v) for k, v in z.items()}


def latent_paths(root: str, limit: int | None = None, rank: int = 0, world_size: int = 1) -> list[Path]:
    """Select paths before loading files so DDP ranks do not duplicate RAM."""
    paths = sorted(Path(root).glob("*.pt")); paths = paths[:limit] if limit else paths
    return paths[rank::world_size]


def latent_refs(root: str, target: dict, limit: int | None = None, rank: int = 0, world_size: int = 1) -> list[tuple[Path, int]]:
    lookup = {(int(s), int(w)): i for i, (s, w) in enumerate(zip(target["scene_id"], target["window_start"]))}
    refs = []; skipped = 0
    for path in latent_paths(root, limit, rank, world_size):
        row = torch.load(path, map_location="cpu", weights_only=False)
        key = (int(row["scene_id"]), int(row["window_start"]))
        # Some historical CLEVRER scenes have fewer valid windows in targets
        # than in the latent cache (for example a short final window). Ignore
        # those cache records; training/evaluation must use aligned pairs.
        if key not in lookup:
            skipped += 1
            continue
        refs.append((path, lookup[key]))
        del row
    if skipped:
        warnings.warn(f"skipped {skipped} latent files without matching target (root={root})", RuntimeWarning)
    return refs


def targets(target: dict, index: int) -> dict[str, torch.Tensor]:
    state, valid = target["state"][index].float(), target["state_valid"][index].bool()
    velocity = state[..., 5:7]; speed = velocity.norm(dim=-1)
    av = valid[:, 1:] & valid[:, :-1]
    acceleration = (velocity[:, 1:] - velocity[:, :-1]).norm(dim=-1)
    speed_v, acc_v = speed[valid], acceleration[av]
    first = target["first_contact_class"][index].long(); contact_ok = first.ge(0)
    contact_event = (first < STEPS) & contact_ok
    ttc = first.float() / (STEPS - 1)
    return {"speed": speed_v.mean() if speed_v.numel() else torch.tensor(float("nan")),
            "acceleration": acc_v.mean() if acc_v.numel() else torch.tensor(float("nan")),
            "ttc": ttc[contact_event].min() if contact_event.any() else torch.tensor(float("nan")),
            "contact_event": contact_event.any().float()}


def fit_regression(x, y, tx, ty, epochs=120):
    x, y, tx, ty = [torch.as_tensor(v).float() for v in (x, y, tx, ty)]
    m, s = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-5); x, tx = (x-m)/s, (tx-m)/s
    h = torch.nn.Linear(x.size(1), 1); opt = torch.optim.AdamW(h.parameters(), lr=.02, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True); loss = torch.nn.functional.smooth_l1_loss(h(x).squeeze(-1), y); loss.backward(); opt.step()
    with torch.no_grad(): pred = h(tx).squeeze(-1)
    return {"mae": float((pred-ty).abs().mean()), "rmse": float((pred-ty).square().mean().sqrt()), "r2": float(1-(pred-ty).square().sum()/((ty-ty.mean()).square().sum().clamp_min(1e-8))), "samples": len(ty)}


def fit_binary(x, y, tx, ty, epochs=120):
    x, y, tx, ty = [torch.as_tensor(v).float() for v in (x, y, tx, ty)]
    m, s = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-5); x, tx = (x-m)/s, (tx-m)/s
    h = torch.nn.Linear(x.size(1), 1); opt = torch.optim.AdamW(h.parameters(), lr=.02, weight_decay=1e-4); pos=y.sum().clamp_min(1); pw=((len(y)-pos)/pos).clamp(max=50)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True); loss=torch.nn.functional.binary_cross_entropy_with_logits(h(x).squeeze(-1),y,pos_weight=pw); loss.backward(); opt.step()
    with torch.no_grad(): score=h(tx).squeeze(-1)
    t=ty>.5; p,n=int(t.sum()),int((~t).sum()); auc=None if not p or not n else float(((score.argsort().argsort().float()+1)[t].sum()-p*(p+1)/2)/(p*n))
    return {"auroc":auc,"accuracy":float(((score.sigmoid()>.5)==t).float().mean()),"samples":len(ty)}


def encode(refs, target, checkpoint, device, limit=None):
    LOGGER.info("encode start: refs=%d device=%s checkpoint=%s", len(refs if limit is None else refs[:limit]), device, checkpoint)
    model=CLEVRERDecoder().to(device); payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    state = payload.get("model", payload)
    if "decoder" in payload and "clevrer_probes" in payload:
        state = {f"decoder.{k}": v for k, v in payload["decoder"].items()}; state.update({f"probes.{k}": v for k, v in payload["clevrer_probes"].items()})
    model.load_state_dict(state,strict=True); model.eval(); feats=[]; ys=[]
    with torch.inference_mode():
        for path, target_index in refs[:limit] if limit else refs:
            row = torch.load(path, map_location="cpu", weights_only=False)
            out=model(row["context_tokens"].unsqueeze(0).to(device).float(),row["future_tokens"].unsqueeze(0).to(device).float(),return_features=True)
            dynamic=out["time_features"][0].flatten().cpu(); relation=out["pair_time_features"][0].flatten().cpu(); entity=out["object_tokens"][0].flatten().cpu()
            feats.append({"entity":entity,"dynamic":dynamic,"relation":relation,"full":torch.cat((entity,dynamic,relation)).cpu()}); ys.append(targets(target,int(target_index)))
            del row, out
            if len(feats) % 1000 == 0: LOGGER.info("encode progress: %d samples", len(feats))
    LOGGER.info("encode done: samples=%d feature_shapes=%s", len(feats), {k: tuple(v.shape) for k,v in feats[0].items()} if feats else {})
    return feats, ys


def decode(train_f, train_y, test_f, test_y):
    LOGGER.info("decode start: train=%d validation=%d", len(train_f), len(test_f))
    result={}
    for name in ("speed","acceleration","ttc"):
        result[name]={}; kt=torch.tensor([torch.isfinite(y[name]) for y in train_y]); kv=torch.tensor([torch.isfinite(y[name]) for y in test_y])
        for layer in ("entity","dynamic","relation","full"):
            LOGGER.info("fit regression: target=%s feature=%s train=%d test=%d dim=%d", name, layer, int(kt.sum()), int(kv.sum()), train_f[0][layer].numel())
            x=torch.stack([f[layer] for f,k in zip(train_f,kt) if k]); tx=torch.stack([f[layer] for f,k in zip(test_f,kv) if k]); y=torch.stack([v[name] for v,k in zip(train_y,kt) if k]); ty=torch.stack([v[name] for v,k in zip(test_y,kv) if k]); result[name][layer]=fit_regression(x,y,tx,ty)
    result["contact_event"]={}; y=torch.stack([v["contact_event"] for v in train_y]); ty=torch.stack([v["contact_event"] for v in test_y])
    for layer in ("entity","dynamic","relation","full"):
        LOGGER.info("fit binary: target=contact_event feature=%s train=%d test=%d dim=%d", layer, len(y), len(ty), train_f[0][layer].numel())
        result["contact_event"][layer]=fit_binary(torch.stack([f[layer] for f in train_f]),y,torch.stack([f[layer] for f in test_f]),ty)
    LOGGER.info("decode done")
    return result


def intervention(context, future, mode, seed=239):
    if mode=="baseline": return context,future
    if mode=="future_reverse": return context,future.flip(1)
    if mode=="future_static": return context,future.mean(1,keepdim=True).expand_as(future).clone()
    if mode=="future_repeat": return context,future[:,:1].expand_as(future).clone()
    if mode=="future_shuffle":
        g=torch.Generator(device=future.device).manual_seed(seed); return context,future[:,torch.randperm(future.size(1),generator=g,device=future.device)]
    raise ValueError(mode)


def intervention_metrics(out, target, index):
    state,valid=target["state"][index].float(),target["state_valid"][index].bool(); pred=out["trajectory_2d"][0].float().cpu(); n=int(target["object_present"][index].sum()); costs=np.zeros((OBJECT_COUNT,n))
    for i in range(OBJECT_COUNT):
        for j in range(n): costs[i,j]=float((pred[i]-state[j]).abs()[valid[j]].mean()) if valid[j].any() else 1e9
    slots,cols=linear_sum_assignment(costs); se=[]; ae=[]
    for i,j in zip(slots,cols):
        m=valid[j]; se.append(float((pred[i,:,5:7].norm(dim=-1)-state[j,:,5:7].norm(dim=-1))[m].abs().mean())); q=m[1:]&m[:-1]
        if q.any(): ae.append(float(((pred[i,1:,5:7]-pred[i,:-1,5:7]).norm(dim=-1)-(state[j,1:,5:7]-state[j,:-1,5:7]).norm(dim=-1))[q].abs().mean()))
    return {"speed_mae":float(np.mean(se)) if se else None,"acceleration_mae":float(np.mean(ae)) if ae else None}


def run_interventions(refs, target, checkpoint, device, limit=None):
    LOGGER.info("intervention start: refs=%d device=%s", len(refs if limit is None else refs[:limit]), device)
    model=CLEVRERDecoder().to(device); payload=torch.load(checkpoint,map_location="cpu",weights_only=False); state=payload.get("model",payload)
    if "decoder" in payload and "clevrer_probes" in payload:
        state={f"decoder.{k}":v for k,v in payload["decoder"].items()}; state.update({f"probes.{k}":v for k,v in payload["clevrer_probes"].items()})
    model.load_state_dict(state,strict=True); model.eval(); modes=("baseline","future_reverse","future_shuffle","future_static","future_repeat"); allm={m:[] for m in modes}
    with torch.inference_mode():
        for path, target_index in refs[:limit] if limit else refs:
            row = torch.load(path, map_location="cpu", weights_only=False)
            c=row["context_tokens"].unsqueeze(0).to(device).float(); f=row["future_tokens"].unsqueeze(0).to(device).float()
            for m in modes:
                x,y=intervention(c,f,m); allm[m].append(intervention_metrics(model(x,y),target,int(target_index)))
            del row
    selected = refs if limit is None else refs[:limit]
    summary={m:{k:(float(np.mean([v[k] for v in vals if v[k] is not None])) if any(v[k] is not None for v in vals) else None) for k in ("speed_mae","acceleration_mae")} for m,vals in allm.items()}; base=summary["baseline"]; delta={m:{k:(None if summary[m][k] is None or base[k] is None else summary[m][k]-base[k]) for k in base} for m in modes if m!="baseline"}; LOGGER.info("intervention done: %s", summary); return {"samples":len(selected),"summary":summary,"delta_vs_baseline":delta}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [pid=%(process)d] %(message)s")
    ctx=task_context();p=argparse.ArgumentParser(); p.add_argument("--train-latents",required=True); p.add_argument("--validation-latents",required=True); p.add_argument("--train-targets",required=True); p.add_argument("--validation-targets",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--output",required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--max-samples",type=int); p.add_argument("--skip-interventions",action="store_true"); p.add_argument("--rank",type=int,default=0); p.add_argument("--world-size",type=int,default=1); p.add_argument("--shard-output"); p.add_argument("--merge-shards",action="store_true"); apply_cli_defaults(p,ctx);a=p.parse_args(); a.rank=int(os.environ.get("RANK",a.rank)); a.world_size=int(os.environ.get("WORLD_SIZE",a.world_size));
    if a.world_size > 1 and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", a.rank))
        a.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    torch.manual_seed(239+a.rank); np.random.seed(239+a.rank)
    LOGGER.info("start: rank=%d world_size=%d device=%s max_samples=%s", a.rank, a.world_size, a.device, a.max_samples or "all")
    trt=load_targets(a.train_targets,a.max_samples); vat=load_targets(a.validation_targets,a.max_samples); LOGGER.info("targets loaded: train=%d validation=%d", len(trt["scene_id"]), len(vat["scene_id"]))
    tr=latent_refs(a.train_latents,trt,a.max_samples,a.rank,a.world_size); va=latent_refs(a.validation_latents,vat,a.max_samples,a.rank,a.world_size); LOGGER.info("aligned refs: train=%d validation=%d", len(tr), len(va))
    if a.world_size > 1 and not a.merge_shards:
        if not a.shard_output: p.error("--shard-output is required when --world-size > 1")
        tr_local, va_local = tr, va
        tf,ty=encode(tr_local,trt,a.checkpoint,a.device); vf,vy=encode(va_local,vat,a.checkpoint,a.device)
        shard=Path(a.shard_output); shard.mkdir(parents=True,exist_ok=True)
        torch.save({"features":tf,"targets":ty}, shard/f"train_rank{a.rank:02d}.pt")
        torch.save({"features":vf,"targets":vy}, shard/f"validation_rank{a.rank:02d}.pt")
        if not a.skip_interventions:
            torch.save(run_interventions(va_local, vat, a.checkpoint, a.device), shard/f"interventions_rank{a.rank:02d}.pt")
        LOGGER.info("shard written: train=%d validation=%d path=%s", len(tf), len(vf), shard); print(json.dumps({"rank":a.rank,"world_size":a.world_size,"train":len(tf),"validation":len(vf)})); return
    if a.merge_shards:
        if not a.shard_output: p.error("--shard-output is required with --merge-shards")
        shard=Path(a.shard_output); packs=[torch.load(shard/f"train_rank{i:02d}.pt",map_location="cpu",weights_only=False) for i in range(a.world_size)]; vals=[torch.load(shard/f"validation_rank{i:02d}.pt",map_location="cpu",weights_only=False) for i in range(a.world_size)]
        tf=sum((x["features"] for x in packs),[]); ty=sum((x["targets"] for x in packs),[]); vf=sum((x["features"] for x in vals),[]); vy=sum((x["targets"] for x in vals),[])
        result={"protocol":"clevrer_dynamic_selectivity_v1","train_samples":len(tf),"validation_samples":len(vf),"decode":decode(tf,ty,vf,vy)}
        if not a.skip_interventions:
            parts=[torch.load(shard/f"interventions_rank{i:02d}.pt",map_location="cpu",weights_only=False) for i in range(a.world_size)]
            total=sum(int(x["samples"]) for x in parts); summary={}
            for mode in parts[0]["summary"]:
                summary[mode]={k:float(sum(x["summary"][mode][k]*x["samples"] for x in parts)/total) for k in ("speed_mae","acceleration_mae")}
            base=summary["baseline"]; result["interventions"]={"samples":total,"summary":summary,"delta_vs_baseline":{m:{k:summary[m][k]-base[k] for k in base} for m in summary if m!="baseline"}}
        out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+"\n"); LOGGER.info("final metrics written: %s", out); print(json.dumps(result,indent=2)); return
    tf,ty=encode(tr,trt,a.checkpoint,a.device,a.max_samples); vf,vy=encode(va,vat,a.checkpoint,a.device,a.max_samples); result={"protocol":"clevrer_dynamic_selectivity_v1","train_samples":len(tf),"validation_samples":len(vf),"decode":decode(tf,ty,vf,vy)}
    if not a.skip_interventions: result["interventions"]=run_interventions(va,vat,a.checkpoint,a.device,a.max_samples)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
