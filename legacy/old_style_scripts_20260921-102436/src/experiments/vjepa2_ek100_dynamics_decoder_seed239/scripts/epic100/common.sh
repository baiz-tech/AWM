#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
experiment=vjepa2_naive_probe_v5_decoder_ek100
run_name=${RUN_NAME:-epic100_official_vjepa2_v5_decoder_seed239_v1}
config=${CONFIG:-recipe/$experiment/configs/epic100-v5-decoder-8gpu.yaml}
python_bin=${PYTHON_BIN:-/data/shared/envs/vjepa2-312/bin/python}
conda_sh=${CONDA_SH:-/home/shyang/anaconda3/etc/profile.d/conda.sh}
conda_env=${CONDA_ENV:-vjepa2-312}
processed_root=${EPIC_PROCESSED_ROOT:-/data/shared/datasets/EPIC-KITCHENS-processed_v1}
visor_root=${VISOR_PROCESSED_ROOT:-/data/shared/datasets/EPIC-KITCHENS-VISOR-processed_v1}
train_annotations=${TRAIN_ANNOTATIONS:-$processed_root/annotations/EPIC_100_train.csv}
validation_annotations=${VALIDATION_ANNOTATIONS:-$processed_root/annotations/EPIC_100_validation.csv}
data_experiment_root=${DATA_EXPERIMENT_ROOT:-/data/shyang/outputs/vjepa2-baiz/$experiment}
workspace_experiment_root=${WORKSPACE_EXPERIMENT_ROOT:-$repo_root/outputs/runs/$experiment}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29750}

resolve_output_paths() {
  case "$output_mode" in
    both) artifact_run=$data_experiment_root/$run_name; log_run=$workspace_experiment_root/$run_name ;;
    data) artifact_run=$data_experiment_root/$run_name; log_run=$artifact_run ;;
    workspace) artifact_run=$workspace_experiment_root/$run_name; log_run=$artifact_run ;;
    *) echo "invalid output mode: $output_mode" >&2; return 2 ;;
  esac
  train_targets=$artifact_run/targets/train.pt
  validation_targets=$artifact_run/targets/validation.pt
  train_cache=$artifact_run/cache/train
  validation_cache=$artifact_run/cache/validation
  checkpoint_dir=$artifact_run/checkpoints
}
