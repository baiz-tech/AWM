#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
experiment=vjepa2_naive_probe_v5_decoder_ek100
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

output_root=$repo_root/outputs/runs/$experiment/target_visualizations
targets=$output_root/targets_train_visualization.pt
run_dir=$output_root/seed_$seed
output=$run_dir/current_future_gt.mp4

mkdir -p "$output_root" "$run_dir"
if [[ ! -f "$targets" ]]; then
  python -m recipe.$experiment.scripts.epic100.prepare_targets \
    --processed-root /data/shared/datasets/EPIC-KITCHENS-processed_v1 \
    --visor-root /data/shared/datasets/EPIC-KITCHENS-VISOR-processed_v1 \
    --annotations /data/shared/datasets/EPIC-KITCHENS-processed_v1/annotations/EPIC_100_train.csv \
    --train-annotations /data/shared/datasets/EPIC-KITCHENS-processed_v1/annotations/EPIC_100_train.csv \
    --split train \
    --output "$targets"
fi

python -m recipe.$experiment.scripts.epic100.visualize_targets \
  --targets "$targets" \
  --seed "$seed" \
  --output "$output"

echo "video=$output"
