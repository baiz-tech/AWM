#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
config=${CONFIG:?set CONFIG=...}; predictor=${PREDICTOR_CHECKPOINT:?set PREDICTOR_CHECKPOINT=...}; split=${SPLIT:?set SPLIT=data_v1 or readout_data_v1}; output_dir=${OUTPUT_DIR:?set OUTPUT_DIR=...}; num_gpus=${NUM_GPUS:-8}; batch_size=${BATCH_SIZE:-1}; num_workers=${NUM_WORKERS:-2}; python_bin=${PYTHON_BIN:-python}
mkdir -p "$output_dir/$split"; pids=()
for rank in $(seq 0 $((num_gpus-1))); do CUDA_VISIBLE_DEVICES="$rank" "$python_bin" -m src.experiments.vjepa2_physionpp_dynamics_decoder_seed239.scripts.physionpp.cache_full_latents --config "$config" --predictor-checkpoint "$predictor" --split "$split" --output-dir "$output_dir" --batch-size "$batch_size" --num-workers "$num_workers" --rank "$rank" --world-size "$num_gpus" --resume >"$output_dir/$split/rank_${rank}.log" 2>&1 & pids+=("$!"); done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ $status -ne 0 ]]; then echo "worker failed; inspect $output_dir/$split/rank_*.log" >&2; exit 1; fi
"$python_bin" -m src.experiments.vjepa2_physionpp_dynamics_decoder_seed239.scripts.physionpp.cache_full_latents --config "$config" --predictor-checkpoint "$predictor" --split "$split" --output-dir "$output_dir" --manifest-only
