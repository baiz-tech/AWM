#!/usr/bin/env bash
set -euo pipefail
cd /home/zqliu/Work-Space/ABDUCTIVE-WORLD/src.experiments
exec torchrun --standalone --nproc_per_node=8 --master_port=29626 -m object_slot_generalization_v3.train \
 --physion /data/shyang/outputs/ABDUCTIVE-WORLD/vjepa2_naive_probe_physionpp_v1/physionpp_fullpatch_structured_ocp_ddp8_seed239_v5/cache/train \
 --ektrain /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v23_ek100_only/seed239_parallel_v23/cache/train \
 --ekval /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v23_ek100_only/seed239_parallel_v23/cache/probe_validation \
 --out /data/shyang/outputs/ABDUCTIVE-WORLD/object_slot_generalization_v3/hungarian_seed239 --epochs 20 --batch 8
