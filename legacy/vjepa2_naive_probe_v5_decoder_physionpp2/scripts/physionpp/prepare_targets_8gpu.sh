#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dataset_root=${DATASET_ROOT:-/data/ABDUCTIVE-WORLD/physion_v2/extracted}
split=${SPLIT:-data_v1}
output=${OUTPUT:?set OUTPUT=/path/to/targets.pt}
workers=${NUM_GPUS:-8}
work_dir=${WORK_DIR:-${output}.shards}
python_bin=${PYTHON_BIN:-python}
mkdir -p "$work_dir"
pids=()
for rank in $(seq 0 $((workers-1))); do
  CUDA_VISIBLE_DEVICES="$rank" "$python_bin" -m recipe.vjepa2_naive_probe_v5_decoder_physionpp2.scripts.physionpp.prepare_targets --dataset-root "$dataset_root" --split "$split" --output "$work_dir/rank_${rank}.pt" --rank "$rank" --world-size "$workers" --skip-missing-annotations >"$work_dir/rank_${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ $status -ne 0 ]]; then echo "one or more target workers failed; inspect $work_dir/*.log" >&2; exit 1; fi
"$python_bin" -m recipe.vjepa2_naive_probe_v5_decoder_physionpp2.scripts.physionpp.merge_targets --output "$output" --split "$split" --shard-dir "$work_dir"
