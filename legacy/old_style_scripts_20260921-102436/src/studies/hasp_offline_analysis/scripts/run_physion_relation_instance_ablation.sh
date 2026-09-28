#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
RUN=${RUN:-/data/shyang/outputs/ABDUCTIVE-WORLD/reproduce_v12_physion_only/seed239_parallel_v12}
MODE=${OUTPUT_MODE:-both}
NAME=${RUN_NAME:-physion_relation_instance_ablation_v2}
GPUID=${GPU:-0}
MAX_SAMPLES=${MAX_SAMPLES:-}
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=$2; shift 2;;
    --output-mode) MODE=$2; shift 2;;
    --run-name) NAME=$2; shift 2;;
    --gpu) GPUID=$2; shift 2;;
    --max-samples) MAX_SAMPLES=$2; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) echo "usage: $0 [--run PATH] [--output-mode both|data|workspace] [--run-name NAME] [--gpu ID] [--max-samples N] [--dry-run]"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
case "$MODE" in both|data|workspace) ;; *) echo "invalid --output-mode=$MODE" >&2; exit 2;; esac
if [[ "$MODE" == workspace || "$MODE" == both ]]; then WORK_OUT="$ROOT/outputs/runs/src.experiments/$NAME"; fi
if [[ "$MODE" == data || "$MODE" == both ]]; then DATA_OUT="/data/shyang/outputs/vjepa2-baiz/src.experiments/$NAME"; fi
OUT=${OUT:-${DATA_OUT:-$WORK_OUT}}
LOG_OUT=${WORK_OUT:-$OUT}
RUN_CACHE="$RUN/cache/validation"; PROBE="$RUN/probe/best.pt"; PREDICTOR="$RUN/predictor/best.pt"; PRED_CFG="$ROOT/recipe/src.experiments/reproduce_v12_physion_only/predictor_config.yaml"
echo "protocol=physion_true_instance_relation_pair_ablation_v2"; echo "run=$RUN"; echo "output_mode=$MODE"; echo "output=$OUT"; echo "gpu=$GPUID"
[[ "$DRY_RUN" -eq 1 ]] && exit 0
for path in "$RUN_CACHE" "$PROBE" "$PREDICTOR" "$PRED_CFG"; do [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }; done
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite existing output: $OUT" >&2; exit 1; }
[[ -z "${WORK_OUT:-}" || ! -e "$WORK_OUT" ]] || { echo "refusing to overwrite existing workspace output: $WORK_OUT" >&2; exit 1; }
mkdir -p "$OUT" "$LOG_OUT"
args=(--cache "$RUN_CACHE" --probe "$PROBE" --predictor "$PREDICTOR" --predictor-config "$PRED_CFG" --output "$OUT/metrics.json" --device cuda:0)
[[ -n "$MAX_SAMPLES" ]] && args+=(--max-samples "$MAX_SAMPLES")
LOG="$LOG_OUT/analysis.log"
CMD="cd '$ROOT'; CUDA_VISIBLE_DEVICES='$GPUID' PYTHONPATH='$ROOT' python -m src.studies.hasp_offline_analysis.physion_relation_instance_ablation ${args[*]}"
if [[ "$MODE" == both ]]; then CMD+="; cp '$OUT/metrics.json' '$WORK_OUT/metrics.json'"; fi
setsid nohup bash -lc "$CMD" >"$LOG" 2>&1 < /dev/null &
echo $! > "$LOG_OUT/run.pid"
printf 'started pid=%s log=%s\n' "$!" "$LOG"
