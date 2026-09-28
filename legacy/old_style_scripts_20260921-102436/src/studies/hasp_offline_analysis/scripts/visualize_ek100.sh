#!/usr/bin/env bash
set -euo pipefail
cd /home/zqliu/Work-Space/ABDUCTIVE-WORLD
python -m src.studies.hasp_offline_analysis.visualize_ek100 \
  --cache /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v23_ek100_only/seed239_parallel_v23/cache/probe_validation \
  --readout /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v23_ek100_only/seed239_parallel_v23/readout/best.pt \
  --adapter /data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v23_ek100_only/seed239_parallel_v23/adapter/best.pt \
  --annotations /data/shared/datasets/EPIC-KITCHENS-processed_v1/annotations/EPIC_100_train.csv \
  --video-root /data/shared/datasets/EPIC-KITCHENS-processed_v1 \
  --output src.experiments/analyze/results/full/ek100_visualizations \
  --count 8 --device cuda:0
