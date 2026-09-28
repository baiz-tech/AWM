#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
run_dir=${RUN_DIR:-/data/shyang/outputs/vjepa2-baiz/vjepa2_naive_probe_v5_decoder_ek100/epic100_official_vjepa2_v5_decoder_seed239_data_v1}
seed=239
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) seed=$2; shift 2 ;;
    --seed=*) seed=${1#*=}; shift ;;
    *) echo "unsupported argument: $1" >&2; exit 2 ;;
  esac
done
source /home/shyang/anaconda3/etc/profile.d/conda.sh
conda activate vjepa2-312
cd "$repo_root"
output_dir=$repo_root/outputs/runs/vjepa2_naive_probe_v5_decoder_ek100/prediction_visualizations/seed_$seed
mkdir -p "$output_dir"
python -m src.experiments.vjepa2_ek100_dynamics_decoder_seed239.scripts.epic100.visualize_predictions \
  --run-dir "$run_dir" --seed "$seed" --split validation --tap-hand-only \
  --output "$output_dir/visualization.mp4"
echo "gt_video=$output_dir/current_future_gt.mp4"
echo "prediction_video=$output_dir/current_future_pred.mp4"
echo "combined_video=$output_dir/current_future_gt_pred.mp4"
