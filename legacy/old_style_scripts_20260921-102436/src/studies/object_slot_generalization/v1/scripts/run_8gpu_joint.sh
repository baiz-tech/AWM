#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zqliu/Work-Space/ABDUCTIVE-WORLD/src.experiments
PHYSION=/data/shyang/outputs/ABDUCTIVE-WORLD/vjepa2_naive_probe_physionpp_v1/physionpp_fullpatch_structured_ocp_ddp8_seed239_v5/cache/train
EKROOT=/data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v23_ek100_only/seed239_parallel_v23/cache
OUT=/data/shyang/outputs/ABDUCTIVE-WORLD/object_slot_generalization/joint_seed239
mkdir -p "$OUT"
cd "$ROOT"
exec torchrun --standalone --nproc_per_node=8 --master_port=29617 -m object_slot_generalization.train_joint \
  --physion-train "$PHYSION" \
  --ek100-train "$EKROOT/train" \
  --ek100-validation "$EKROOT/probe_validation" \
  --output "$OUT" --epochs 5 --batch-size 8
