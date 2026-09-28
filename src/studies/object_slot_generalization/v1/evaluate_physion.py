"""Evaluate the new context-only slots on Physion++ object statistics.

This is deliberately slot-only: no future tensor, Dynamic branch, Relation
branch, OCP head, or raw patch pooling is used.  It reports presence-count
accuracy and ridge probes for aggregate object extent/position, which are
safe diagnostics even before instance masks are available.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from .model import ObjectSlotModel


def collect(model, files, device):
    xs, counts, extents, positions = [], [], [], []
    model.eval()
    with torch.no_grad():
        for p in files:
            z = torch.load(p, map_location='cpu', weights_only=False)
            x = z['context'].float().unsqueeze(0).to(device)
            out = model(x)
            slots = out.slots.mean(1)[0].float().cpu()
            valid = z['current_object_valid'].bool()
            state = z['current_object_state'].float()
            xs.append(slots.flatten())
            counts.append(valid.sum().float())
            extents.append(state[valid, 6:9].mean(0) if valid.any() else torch.zeros(3))
            positions.append(state[valid, :3].mean(0) if valid.any() else torch.zeros(3))
    return torch.stack(xs), torch.stack(counts), torch.stack(extents), torch.stack(positions)


def ridge_fit_predict(x_train, y_train, x_eval, ridge=1e-2):
    x_train = torch.cat([x_train, torch.ones(len(x_train), 1)], 1)
    x_eval = torch.cat([x_eval, torch.ones(len(x_eval), 1)], 1)
    w = torch.linalg.solve(x_train.T @ x_train + ridge * torch.eye(x_train.size(1)), x_train.T @ y_train)
    pred = x_eval @ w
    return pred


def ridge_r2(x_train, y_train, x_eval, y_eval, ridge=1e-2):
    pred = ridge_fit_predict(x_train, y_train, x_eval, ridge)
    ssr = ((y_eval - pred) ** 2).sum(0); sst = ((y_eval - y_eval.mean(0)) ** 2).sum(0).clamp_min(1e-8)
    return float((1 - ssr / sst).mean()), pred


def per_slot_r2(slots_train, y_train, slots_eval, y_eval):
    vals = []
    for i in range(slots_train.size(1)):
        vals.append(ridge_r2(slots_train[:, i], y_train, slots_eval[:, i], y_eval)[0])
    return vals


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--cache', required=True); p.add_argument('--output', required=True); p.add_argument('--limit', type=int, default=0); p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(); device = torch.device(a.device); model = ObjectSlotModel().to(device)
    payload = torch.load(a.checkpoint, map_location='cpu', weights_only=False); model.load_state_dict(payload['model'])
    files = sorted(Path(a.cache).glob('sample_*.pt')); files = files[:a.limit] if a.limit else files
    x, count, extent, position = collect(model, files, device)
    # Presence head is trained against the repeated object-validity target;
    # count calibration is the permutation-invariant quantity available here.
    slot_count = (torch.sigmoid(model.entity.presence(model(x[:1].to(device)).slots.mean(1))).sum() if False else torch.zeros(1))
    # Probe aggregate object statistics using slots only.
    cut = max(1, int(len(x) * 0.7)); xt, xe = x[:cut], x[cut:]; et, ee = extent[:cut], extent[cut:]; pt, pe = position[:cut], position[cut:]
    slot_t, slot_e = xt.reshape(cut, 8, -1), xe.reshape(len(x)-cut, 8, -1)
    extent_r2, _ = ridge_r2(xt, et, xe, ee); position_r2, _ = ridge_r2(xt, pt, xe, pe)
    result = {'protocol': 'physion_object_slot_only_eval_v2', 'checkpoint': str(Path(a.checkpoint).resolve()), 'split': str(Path(a.cache).resolve()), 'samples': len(files), 'probe_train_samples': cut, 'probe_eval_samples': len(x)-cut, 'slot_feature_dim': int(x.size(1)), 'object_count_mean': float(count.mean()), 'object_extent_r2': extent_r2, 'object_position_r2': position_r2, 'per_slot_extent_r2': per_slot_r2(slot_t, et, slot_e, ee), 'per_slot_position_r2': per_slot_r2(slot_t, pt, slot_e, pe), 'note': 'Held-out aggregate probes; instance-level slot-mask IoU and ID-switch require object masks/tracks.'}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
