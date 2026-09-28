#!/usr/bin/env bash
set -euo pipefail

# End-to-end Physion++ Dynamic selectivity analysis.
# Feature export is sharded over 8 GPUs; compact analysis runs once after.

ROOT=$(cd "$(dirname "$0")/../../../../" && pwd)
RUN=${RUN:-/data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v12_physion_only/seed239_parallel_v12}
OUT=${OUT:-/data/shyang/outputs/vjepa2-baiz/analyses/physion_dynamic_selectivity_v1}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
WORLD_SIZE=${WORLD_SIZE:-8}
SMOKE_SAMPLES=${SMOKE_SAMPLES:-16}
MAX_SAMPLES=${MAX_SAMPLES:-}
DEVICE=${DEVICE:-cuda:0}
SKIP_SMOKE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: run_physion_dynamic_selectivity.sh [options]
  --run PATH             Existing Physion++ run root
  --output PATH          Analysis output root
  --gpus LIST            CUDA devices, default 0,1,2,3,4,5,6,7
  --world-size N         Number of shards, default 8
  --max-samples N        Limit full export/analysis (optional)
  --smoke-samples N      Smoke-test samples, default 16
  --skip-smoke           Skip preliminary smoke test
  --dry-run              Print resolved paths only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=$2; shift 2;;
    --output) OUT=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;;
    --world-size) WORLD_SIZE=$2; shift 2;;
    --max-samples) MAX_SAMPLES=$2; shift 2;;
    --smoke-samples) SMOKE_SAMPLES=$2; shift 2;;
    --skip-smoke) SKIP_SMOKE=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [[ "${#GPU_LIST[@]}" -ne "$WORLD_SIZE" ]]; then
  echo "--gpus count (${#GPU_LIST[@]}) must equal --world-size ($WORLD_SIZE)" >&2
  exit 2
fi

TRAIN_CACHE="$RUN/cache/train"
VALIDATION_CACHE="$RUN/cache/validation"
CHECKPOINT="$RUN/probe/best.pt"
TRAIN_FEATURES="$OUT/features_train"
VALIDATION_FEATURES="$OUT/features_validation"
SMOKE_TRAIN_FEATURES="$OUT/features_train_smoke"
SMOKE_VALIDATION_FEATURES="$OUT/features_validation_smoke"

for path in "$TRAIN_CACHE" "$VALIDATION_CACHE" "$CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done

echo "protocol=physion_dynamic_selectivity_v1"
echo "root=$ROOT"
echo "run=$RUN"
echo "output=$OUT"
echo "gpus=$GPUS"
echo "world_size=$WORLD_SIZE"
echo "max_samples=${MAX_SAMPLES:-all}"

[[ "$DRY_RUN" -eq 1 ]] && exit 0
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite existing output: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_export() {
  local cache=$1 output=$2 tag=$3 limit=$4
  mkdir -p "$output"
  local pids=()
  for rank in "${!GPU_LIST[@]}"; do
    local args=(--cache "$cache" --checkpoint "$CHECKPOINT" --output "$output" --device cuda:0 --rank "$rank" --world-size "$WORLD_SIZE")
    [[ -n "$limit" ]] && args+=(--max-samples "$limit")
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$rank]}" python -m src.studies.hasp_offline_analysis.extract_physion "${args[@]}" >"$OUT/${tag}_rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]] || { echo "$tag export failed; inspect $OUT/${tag}_rank*.log" >&2; exit 1; }
}

if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  run_export "$TRAIN_CACHE" "$SMOKE_TRAIN_FEATURES" train_smoke "$SMOKE_SAMPLES"
  run_export "$VALIDATION_CACHE" "$SMOKE_VALIDATION_FEATURES" validation_smoke "$SMOKE_SAMPLES"
  python -m src.studies.hasp_offline_analysis.physion_dynamic_selectivity \
    --train-features "$SMOKE_TRAIN_FEATURES" --test-features "$SMOKE_VALIDATION_FEATURES" \
    --cache "$VALIDATION_CACHE" --checkpoint "$CHECKPOINT" --device "$DEVICE" \
    --max-samples "$SMOKE_SAMPLES" --output "$OUT/smoke_metrics.json" >"$OUT/smoke_analysis.log" 2>&1
fi

run_export "$TRAIN_CACHE" "$TRAIN_FEATURES" train "${MAX_SAMPLES:-}"
run_export "$VALIDATION_CACHE" "$VALIDATION_FEATURES" validation "${MAX_SAMPLES:-}"

analysis_args=(--train-features "$TRAIN_FEATURES" --test-features "$VALIDATION_FEATURES" --cache "$VALIDATION_CACHE" --checkpoint "$CHECKPOINT" --device "$DEVICE" --output "$OUT/metrics.json")
[[ -n "$MAX_SAMPLES" ]] && analysis_args+=(--max-samples "$MAX_SAMPLES")
python -m src.studies.hasp_offline_analysis.physion_dynamic_selectivity "${analysis_args[@]}" >"$OUT/analysis.log" 2>&1

python - <<PY
import json
from pathlib import Path
Path("$OUT/manifest.json").write_text(json.dumps({
  "protocol": "physion_dynamic_selectivity_v1", "run": str(Path("$RUN").resolve()),
  "output": str(Path("$OUT").resolve()), "gpus": "$GPUS", "world_size": $WORLD_SIZE,
  "smoke": bool($((1-SKIP_SMOKE))), "max_samples": "${MAX_SAMPLES:-all}"
}, indent=2) + "\n")
PY
echo "completed: $OUT/metrics.json"
