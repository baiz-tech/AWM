#!/usr/bin/env bash
set -euo pipefail
cd /home/zqliu/Work-Space/ABDUCTIVE-WORLD
CUDA_VISIBLE_DEVICES=0 python -m src.studies.hasp_offline_analysis.physion_predictor_regen_ablation \
  --cache /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v12_physion_only/seed239_parallel_v12/cache/validation \
  --probe /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v12_physion_only/seed239_parallel_v12/probe/best.pt \
  --predictor /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v12_physion_only/seed239_parallel_v12/predictor/best.pt \
  --predictor-config src.experiments/reproduce_v12_physion_only/predictor_config.yaml \
  --output src.experiments/analyze/results/full/metrics/physion_predictor_regen_ablation \
  --batch-size 2 --ratio 0.5 --device cuda:0
