#!/usr/bin/env bash
set -euo pipefail

cd /home/zqliu/Work-Space/ABDUCTIVE-WORLD
cache=/data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v12_physion_only/seed239_parallel_v12/cache/validation
checkpoint=/data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v12_physion_only/seed239_parallel_v12/probe/best.pt
output=src.experiments/analyze/results/full/metrics/physion_input_ablation_ratio50

python -m src.studies.hasp_offline_analysis.physion_input_ablation \
  --cache "$cache" \
  --checkpoint "$checkpoint" \
  --output "$output" \
  --ratio 0.50 \
  --device cuda:0
