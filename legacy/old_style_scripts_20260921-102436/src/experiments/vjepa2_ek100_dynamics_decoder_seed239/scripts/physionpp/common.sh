#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
experiment=vjepa2_naive_probe_v5_decoder_physionpp2
run_name=${RUN_NAME:-physionpp_segmentation_2d_decoder_seed239_v1}
config=${CONFIG:-recipe/$experiment/configs/physionpp-fullpatch-structured-probe-v1-8gpu.yaml}
data_root=${DATA_ROOT:-/data/shyang/outputs/vjepa2-baiz/$experiment}
workspace_root=${WORKSPACE_ROOT:-$repo_root/outputs/runs/$experiment}
output_mode=${OUTPUT_MODE:-both}
case "$output_mode" in
  both) artifact_run=$data_root/$run_name; log_run=$workspace_root/$run_name ;;
  data) artifact_run=$data_root/$run_name; log_run=$artifact_run ;;
  workspace) artifact_run=$workspace_root/$run_name; log_run=$artifact_run ;;
  *) echo "invalid output mode: $output_mode" >&2; exit 2 ;;
esac
predictor=${PREDICTOR_CHECKPOINT:-$repo_root/outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/best.pt}
dataset=${DATASET_ROOT:-/data/ABDUCTIVE-WORLD/physion_v2/extracted}
python_bin=${PYTHON_BIN:-python}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29640}
