#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../../" && pwd)
DATA=${DATA:-/data/shyang/outputs/vjepa2-baiz/reproduce_v1_clevrer_only/clevrer_fullpatch_data_v1}
CHECKPOINT=${CHECKPOINT:-$ROOT/outputs/runs/vjepa2_naive_probe_v5_decoder/clevrer_dynamics_decoder_multiwindow_seed239_retry_v1/best.pt}
OUT=${OUT:-/data/shyang/outputs/vjepa2-baiz/analyses/clevrer_dynamic_selectivity_v1}
DEVICE=${DEVICE:-cuda:0}
MAX_SAMPLES=${MAX_SAMPLES:-}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
WORLD_SIZE=${WORLD_SIZE:-8}
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA=$2; shift 2;;
    --checkpoint) CHECKPOINT=$2; shift 2;;
    --output) OUT=$2; shift 2;;
    --device) DEVICE=$2; shift 2;;
    --max-samples) MAX_SAMPLES=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;;
    --world-size) WORLD_SIZE=$2; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) echo "Usage: $0 [--data PATH] [--checkpoint PATH] [--output PATH] [--device DEVICE] [--gpus LIST] [--world-size N] [--max-samples N] [--dry-run]"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

TRAIN="$DATA/latents/train"; VAL="$DATA/latents/validation"; TT="$DATA/targets_train.pt"; VT="$DATA/targets_validation.pt"
for p in "$TRAIN" "$VAL" "$TT" "$VT" "$CHECKPOINT"; do [[ -e "$p" ]] || { echo "missing input: $p" >&2; exit 1; }; done
echo "protocol=clevrer_dynamic_selectivity_v1"; echo "data=$DATA"; echo "checkpoint=$CHECKPOINT"; echo "output=$OUT"; echo "device=$DEVICE"; echo "gpus=$GPUS"; echo "world_size=$WORLD_SIZE"; echo "max_samples=${MAX_SAMPLES:-all}"
[[ "$DRY_RUN" -eq 1 ]] && exit 0
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite existing output: $OUT" >&2; exit 1; }
mkdir -p "$OUT"; cd "$ROOT"; export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
rm -rf "$OUT/shards"; mkdir -p "$OUT/shards"
args=(--train-latents "$TRAIN" --validation-latents "$VAL" --train-targets "$TT" --validation-targets "$VT" --checkpoint "$CHECKPOINT" --output "$OUT/metrics.json" --world-size "$WORLD_SIZE" --shard-output "$OUT/shards")
[[ -n "$MAX_SAMPLES" ]] && args+=(--max-samples "$MAX_SAMPLES")
CUDA_VISIBLE_DEVICES="$GPUS" torchrun --standalone --nproc_per_node="$WORLD_SIZE" -m src.studies.hasp_offline_analysis.clevrer_dynamic_selectivity "${args[@]}" >"$OUT/export.log" 2>&1
merge_args=(--train-latents "$TRAIN" --validation-latents "$VAL" --train-targets "$TT" --validation-targets "$VT" --checkpoint "$CHECKPOINT" --output "$OUT/metrics.json" --device "$DEVICE" --world-size "$WORLD_SIZE" --shard-output "$OUT/shards" --merge-shards)
[[ -n "$MAX_SAMPLES" ]] && merge_args+=(--max-samples "$MAX_SAMPLES")
python -m src.studies.hasp_offline_analysis.clevrer_dynamic_selectivity "${merge_args[@]}" >"$OUT/analysis.log" 2>&1
python - <<PY
import json
from pathlib import Path
Path("$OUT/manifest.json").write_text(json.dumps({"protocol":"clevrer_dynamic_selectivity_v1","data":str(Path("$DATA").resolve()),"checkpoint":str(Path("$CHECKPOINT").resolve()),"output":str(Path("$OUT").resolve()),"device":"$DEVICE","max_samples":"${MAX_SAMPLES:-all}"},indent=2)+"\n")
PY
echo "completed: $OUT/metrics.json"
