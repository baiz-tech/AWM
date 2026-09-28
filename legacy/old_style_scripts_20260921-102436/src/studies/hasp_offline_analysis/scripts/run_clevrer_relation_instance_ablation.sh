#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/data/shared/envs/vjepa2-312/bin/python}
RUN=${RUN:-/data/shyang/outputs/vjepa2-baiz/reproduce_v1_clevrer_only/clevrer_fullpatch_data_v1}
DATASET_ROOT=${DATASET_ROOT:-/data/shared/datasets/CLEVRER}
TARGETS=${TARGETS:-$RUN/targets_validation.pt}
CACHE_ROOT=${CACHE_ROOT:-$RUN/latents}
PROBE=${PROBE_CHECKPOINT:-$ROOT/outputs/runs/vjepa2_naive_probe_v5_decoder/clevrer_dynamics_decoder_multiwindow_seed239_retry_v1/best.pt}
WORLD=${WORLD_CHECKPOINT:-/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/clevrer_vith_16to16_stride2_8gpu/best.pt}
CONFIG=${CONFIG:-$ROOT/recipe/src.experiments/reproduce_v1_clevrer_only/configs/clevrer-fullpatch-structured-probe-v1-8gpu.yaml}
MODE=${OUTPUT_MODE:-workspace}; NAME=${RUN_NAME:-clevrer_relation_instance_ablation_v2_full}; MAX=${MAX_SCENES:-}; SEEDS=${SEEDS:-239,241,251}; AREA_TOL=${AREA_TOLERANCE:-0.25}; GPU=${GPU:-0}; DRY=0
while [[ $# -gt 0 ]]; do case "$1" in --run) RUN=$2; shift 2;; --dataset-root) DATASET_ROOT=$2; shift 2;; --targets) TARGETS=$2; shift 2;; --cache-root) CACHE_ROOT=$2; shift 2;; --probe) PROBE=$2; shift 2;; --world-checkpoint) WORLD=$2; shift 2;; --config) CONFIG=$2; shift 2;; --output-mode) MODE=$2; shift 2;; --run-name) NAME=$2; shift 2;; --max-scenes) MAX=$2; shift 2;; --seeds) SEEDS=$2; shift 2;; --area-tolerance) AREA_TOL=$2; shift 2;; --gpu) GPU=$2; shift 2;; --dry-run) DRY=1; shift;; *) echo "unknown option: $1" >&2; exit 2;; esac; done
case "$MODE" in workspace) OUT="$ROOT/outputs/runs/src.experiments/$NAME";; data) OUT="/data/shyang/outputs/vjepa2-baiz/src.experiments/$NAME";; both) OUT="/data/shyang/outputs/vjepa2-baiz/src.experiments/$NAME";; *) echo invalid output mode >&2; exit 2;; esac
echo "protocol=clevrer_true_instance_relation_pair_ablation_v2"; echo "output_mode=$MODE"; echo "output=$OUT"; echo "cache=$CACHE_ROOT"; echo "targets=$TARGETS"; echo "seeds=$SEEDS"; echo "area_tolerance=$AREA_TOL"
[[ "$DRY" -eq 1 ]] && exit 0
for p in "$CACHE_ROOT/validation" "$TARGETS" "$PROBE" "$WORLD" "$CONFIG" "$DATASET_ROOT/processed_proposals"; do [[ -e "$p" ]] || { echo "missing required input: $p" >&2; exit 1; }; done
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }; mkdir -p "$OUT"
args=(--cache-root "$CACHE_ROOT" --targets "$TARGETS" --probe "$PROBE" --world-checkpoint "$WORLD" --config "$CONFIG" --dataset-root "$DATASET_ROOT" --output "$OUT/metrics.json" --device cuda:0)
[[ -n "$MAX" ]] && args+=(--max-scenes "$MAX")
args+=(--seeds "$SEEDS" --area-tolerance "$AREA_TOL")
setsid nohup env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PYTHON_BIN" -m src.studies.hasp_offline_analysis.clevrer_relation_instance_ablation "${args[@]}" >"$OUT/analysis.log" 2>&1 < /dev/null & echo $! > "$OUT/run.pid"; echo "started pid=$(cat "$OUT/run.pid") log=$OUT/analysis.log"
