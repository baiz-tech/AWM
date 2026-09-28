#!/usr/bin/env bash
set -euo pipefail
dataset=${DATASET_ROOT:-/data/ABDUCTIVE-WORLD/physion_v2/extracted};split=${SPLIT:-data_v1};output=${OUTPUT:?set OUTPUT};workers=${NUM_GPUS:-8};work=${WORK_DIR:-${output}.shards};python_bin=${PYTHON_BIN:-python};mkdir -p "$work";pids=()
for rank in $(seq 0 $((workers-1)));do CUDA_VISIBLE_DEVICES="$rank" "$python_bin" -m legacy.vjepa2_naive_probe_v5_decoder_physionpp2.scripts.physionpp.prepare_target_only --dataset-root "$dataset" --split "$split" --output "$work/rank_$rank.pt" --rank "$rank" --world-size "$workers" --skip-missing-annotations >"$work/rank_$rank.log" 2>&1 & pids+=("$!");done
status=0;for pid in "${pids[@]}";do wait "$pid"||status=1;done;[[ $status == 0 ]]||{ echo "target worker failed: $work" >&2;exit 1;}
"$python_bin" -m legacy.vjepa2_naive_probe_v5_decoder_physionpp2.scripts.physionpp.merge_target_only --shard-dir "$work" --output "$output" --split "$split"
