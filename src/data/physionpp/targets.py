"""Physion++ PKL to 3D object-trajectory targets; excludes target-zone events."""
from __future__ import annotations
import numpy as np


def _frame(meta, index):
    frames = meta.get("frames", {})
    return next((frames[key] for key in (f"{int(index):04d}", str(int(index)), int(index)) if key in frames), None)


def _ids(meta):
    static = meta["static"]
    rigid = np.asarray(static.get("object_ids", []), dtype=np.int64).reshape(-1)
    deform = np.asarray(static.get("obi_object_ids", []), dtype=np.int64).reshape(-1)
    ids = np.concatenate((rigid, deform))
    if len(ids) != len(np.unique(ids)): raise ValueError("Physion++ object ids must be unique")
    return rigid, deform, ids


def _value(mapping, object_id):
    return next((mapping[key] for key in (str(int(object_id)), int(object_id)) if key in mapping), None)


def _vec(value):
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.size != 3 or not np.isfinite(value).all(): raise ValueError("invalid state vector")
    return value


def _state(meta, index, object_id):
    rigid, deform, _ = _ids(meta); frame = _frame(meta, index)
    if not isinstance(frame, dict) or not isinstance(frame.get("objects"), dict): return None
    objects = frame["objects"]; rigid_row = np.flatnonzero(rigid == object_id)
    try:
        if len(rigid_row):
            row = int(rigid_row[0]); position = _vec(np.asarray(objects.get("center", objects.get("positions"))[row])); velocity = _vec(np.asarray(objects["velocities"])[row])
            extent = np.asarray([np.linalg.norm(_vec(np.asarray(objects[hi])[row]) - _vec(np.asarray(objects[lo])[row])) for hi, lo in (("right", "left"), ("top", "bottom"), ("front", "back"))], dtype=np.float32)
            return np.concatenate((position, velocity, extent)), False
        if object_id not in deform: return None
        positions, velocities = np.asarray(_value(objects.get("particles_positions", {}), object_id), dtype=np.float32).reshape(-1, 3), np.asarray(_value(objects.get("particles_velocities", {}), object_id), dtype=np.float32).reshape(-1, 3)
        if not len(positions) or positions.shape != velocities.shape or not np.isfinite(positions).all() or not np.isfinite(velocities).all(): return None
        return np.concatenate((positions.mean(0), velocities.mean(0), np.maximum(np.quantile(positions, .95, axis=0) - np.quantile(positions, .05, axis=0), 1e-6))), True
    except (IndexError, KeyError, TypeError, ValueError): return None


def _trajectory(meta, indices, max_objects):
    rigid, deform, ids = _ids(meta)
    if len(ids) > max_objects: raise ValueError(f"scene contains {len(ids)} objects > max_objects={max_objects}")
    state, valid = np.zeros((max_objects, len(indices), 9), np.float32), np.zeros((max_objects, len(indices)), bool)
    padded, deformable = np.full(max_objects, -1, np.int64), np.ones(max_objects, bool); padded[:len(ids)] = ids; deformable[:len(ids)] = np.isin(ids, deform)
    for obj_row, object_id in enumerate(ids):
        for time, index in enumerate(indices):
            item = _state(meta, index, int(object_id))
            if item is not None: state[obj_row, time], valid[obj_row, time] = item[0], True
    return state, valid, padded, deformable


def make_targets(meta, current_indices, future_indices, max_objects=8, tubelets=8):
    current, current_valid, ids, deformable = _trajectory(meta, current_indices, max_objects)
    future, future_valid_raw, _, _ = _trajectory(meta, future_indices, max_objects)
    reference_valid, reference = current_valid[:, -1], current[:, -1].copy()
    valid_states = reference[reference_valid]
    if not len(valid_states): raise ValueError("no valid object at final context frame")
    scale = max(float(np.linalg.norm(np.ptp(valid_states[:, :3], axis=0) + valid_states[:, 6:9].max(0))), 1e-3)
    center = valid_states[:, :3].mean(0)
    current[:, -1, :3] = (reference[:, :3] - center) / scale; current[:, -1, 3:] /= scale; current[:, :-1] = 0; current_valid[:, :-1] = False; current[~current_valid] = 0
    future[:, :, :3] = (future[:, :, :3] - reference[:, None, :3]) / scale; future[:, :, 3:] /= scale
    future_valid = future_valid_raw & reference_valid[:, None]; future[~future_valid] = 0
    distance = np.linalg.norm(future[:, None, :, :3] - future[None, :, :, :3], axis=-1).astype(np.float32)
    pair_valid = future_valid[:, None] & future_valid[None, :]; pair_valid &= ~np.eye(max_objects, dtype=bool)[:, :, None]; distance[~pair_valid] = 0
    contact, contact_valid = np.zeros((max_objects, max_objects, tubelets), np.float32), np.zeros((max_objects, max_objects, tubelets), bool)
    zone = meta.get("static", {}).get("zone_id"); eligible = (ids >= 0) & ~deformable
    try: eligible &= ids != int(zone)
    except (TypeError, ValueError): eligible[:] = False
    contact_valid[:] = (eligible[:, None] & eligible[None, :] & ~np.eye(max_objects, dtype=bool))[:, :, None]
    id_to_row, groups = {int(value): row for row, value in enumerate(ids) if value >= 0}, np.asarray(future_indices).reshape(tubelets, -1)
    step = int(np.median(np.diff(future_indices))) if len(future_indices) > 1 else 1
    for time, group in enumerate(groups):
        end = int(groups[time + 1, 0]) if time + 1 < tubelets else int(group[-1]) + step
        for index in range(int(group[0]), max(int(group[0]) + 1, end)):
            pairs = (_frame(meta, index) or {}).get("collisions", {}).get("object_ids", [])
            try: pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
            except (TypeError, ValueError): pairs = np.empty((0, 2), dtype=np.int64)
            for first_id, second_id in pairs:
                first, second = id_to_row.get(int(first_id)), id_to_row.get(int(second_id))
                if first is not None and second is not None and contact_valid[first, second, time]: contact[first, second, time] = contact[second, first, time] = 1
    observed = future_valid.reshape(max_objects, tubelets, -1).all(-1); contact_valid &= observed[:, None] & observed[None, :]; contact[~contact_valid] = 0
    has_contact = contact.any(-1); ttc = np.zeros((max_objects, max_objects), np.float32)
    for first, second in zip(*np.nonzero(has_contact)): ttc[first, second] = np.flatnonzero(contact[first, second])[0] / max(tubelets - 1, 1)
    return {"current_object_state": current[:, -1], "current_object_valid": current_valid[:, -1], "future_object_state": future, "future_object_valid": future_valid, "future_pair_distance": distance, "future_pair_valid": pair_valid, "future_contact": contact, "future_contact_valid": contact_valid, "future_time_to_contact": ttc, "future_time_to_contact_valid": has_contact}
