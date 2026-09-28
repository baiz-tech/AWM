#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
EXPERIMENT=clevrer_relation_controls_v1
DATA=${DATA:-$ROOT/outputs/runs/reproduce_v1_clevrer_only/clevrer_fullpatch_data_v1}
TARGET=${TARGET:-$ROOT/outputs/runs/vjepa2_naive_probe_v5_decoder/clevrer_fullpatch_data_v1}
CHECKPOINT=${CHECKPOINT:-$ROOT/outputs/runs/vjepa2_naive_probe_v5_decoder/clevrer_dynamics_decoder_multiwindow_seed239_retry_v1/best.pt}
DEVICE=${DEVICE:-cuda:0}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29641}
MAX_SAMPLES=${MAX_SAMPLES:-}
OUTPUT_MODE=${OUTPUT_MODE:-both}
DRY_RUN=0
FOREGROUND=0
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA=$2; shift 2;;
    --target) TARGET=$2; shift 2;;
    --checkpoint) CHECKPOINT=$2; shift 2;;
    --device) DEVICE=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;;
    --num-gpus) NUM_GPUS=$2; shift 2;;
    --master-port) MASTER_PORT=$2; shift 2;;
    --max-samples) MAX_SAMPLES=$2; shift 2;;
    --output-mode) OUTPUT_MODE=$2; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --foreground) FOREGROUND=1; shift;;
    --resume) RESUME=1; shift;;
    -h|--help)
      echo "Usage: $0 [--max-samples N] [--gpus LIST] [--num-gpus N] [--master-port PORT] [--output-mode both|data|workspace] [--resume] [--dry-run]"
      exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

case "$OUTPUT_MODE" in
  workspace) ARTIFACT_ROOT="$ROOT/outputs/analyses/$EXPERIMENT"; LOG_ROOT="$ARTIFACT_ROOT";;
  data) ARTIFACT_ROOT="/data/shyang/outputs/vjepa2-baiz/analyses/$EXPERIMENT"; LOG_ROOT="$ARTIFACT_ROOT";;
  both) ARTIFACT_ROOT="/data/shyang/outputs/vjepa2-baiz/analyses/$EXPERIMENT"; LOG_ROOT="$ROOT/outputs/analyses/$EXPERIMENT";;
  *) echo "invalid --output-mode: $OUTPUT_MODE" >&2; exit 2;;
esac

for path in "$DATA/latents/train" "$DATA/latents/validation" "$TARGET/targets_train.pt" "$TARGET/targets_validation.pt" "$CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "missing input: $path" >&2; exit 1; }
done

echo "protocol=clevrer_relation_controls_v1"
echo "output_mode=$OUTPUT_MODE"
echo "artifact_root=$ARTIFACT_ROOT"
echo "data=$DATA"
echo "target=$TARGET"
echo "checkpoint=$CHECKPOINT"
echo "device=$DEVICE"
echo "gpus=$GPUS"
echo "num_gpus=$NUM_GPUS"
echo "master_port=$MASTER_PORT"
echo "max_samples=${MAX_SAMPLES:-all}"
[[ "$DRY_RUN" == 1 ]] && exit 0

if [[ "$FOREGROUND" != 1 ]]; then
  [[ "$RESUME" == 1 || ! -e "$ARTIFACT_ROOT" ]] || { echo "refusing to overwrite existing output: $ARTIFACT_ROOT" >&2; exit 1; }
  mkdir -p "$LOG_ROOT"
  cmd=(bash "$0" --foreground --data "$DATA" --target "$TARGET" --checkpoint "$CHECKPOINT" --device "$DEVICE" --gpus "$GPUS" --num-gpus "$NUM_GPUS" --master-port "$MASTER_PORT" --output-mode "$OUTPUT_MODE")
  [[ "$RESUME" == 1 ]] && cmd+=(--resume)
  [[ -n "$MAX_SAMPLES" ]] && cmd+=(--max-samples "$MAX_SAMPLES")
  nohup "${cmd[@]}" >"$LOG_ROOT/launcher.log" 2>&1 &
  echo "$!" >"$LOG_ROOT/launcher.pid"
  echo "started: pid=$! log=$LOG_ROOT/launcher.log"
  exit 0
fi

[[ "$RESUME" == 1 || ! -e "$ARTIFACT_ROOT" ]] || { echo "refusing to overwrite existing output: $ARTIFACT_ROOT" >&2; exit 1; }
mkdir -p "$ARTIFACT_ROOT" "$LOG_ROOT"
cd "$ROOT"
CONDA_BASE=${CONDA_BASE:-$(conda info --base)}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-vjepa2-312}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
args=(--train-latents "$DATA/latents/train" --validation-latents "$DATA/latents/validation" --train-targets "$TARGET/targets_train.pt" --validation-targets "$TARGET/targets_validation.pt" --checkpoint "$CHECKPOINT" --device "$DEVICE" --output "$ARTIFACT_ROOT/metrics.json")
[[ -n "$MAX_SAMPLES" ]] && args+=(--max-samples "$MAX_SAMPLES")
SHARD_ROOT="$ARTIFACT_ROOT/shards_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SHARD_ROOT"
CUDA_VISIBLE_DEVICES="$GPUS" python -m torch.distributed.run --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$MASTER_PORT" --nproc_per_node="$NUM_GPUS" \
  -m src.studies.clevrer_relation_controls.run "${args[@]}" \
  --world-size "$NUM_GPUS" --shard-output "$SHARD_ROOT" >"$ARTIFACT_ROOT/export.log" 2>&1
python -m src.studies.clevrer_relation_controls.run "${args[@]}" \
  --world-size "$NUM_GPUS" --shard-output "$SHARD_ROOT" --merge-shards >"$ARTIFACT_ROOT/analysis.log" 2>&1
printf '%s\n' "completed: $ARTIFACT_ROOT/metrics.json"
