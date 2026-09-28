#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
parse_common_args "$@"

run_stage() {
  for name in best.pt latest.pt history.json manifest.json; do
    [[ ! -e "$probe_root/$name" ]] || {
      echo "probe artifact exists: $probe_root/$name; set PROBE_RUN_NAME to a new run name" >&2
      return 1
    }
  done
  mkdir -p "$probe_root" "$probe_log_root"
  cd "$repo_root"
  local limit_args=()
  [[ -n ${MAX_TRAIN_SCENES:-} ]] && limit_args+=(--max-train-scenes "$MAX_TRAIN_SCENES")
  [[ -n ${MAX_VALIDATION_SCENES:-} ]] && limit_args+=(--max-validation-scenes "$MAX_VALIDATION_SCENES")
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} "$python_bin" -m torch.distributed.run \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=${TRAIN_PORT:-29712} \
    --nproc_per_node=${NUM_GPUS:-8} "$script_dir/train_structured_probe.py" \
    --nonpredictive-cache-root "$artifact_root/latents_nonpredictive" \
    --predictive-cache-root "$artifact_root/latents_predictive" \
    --train-targets "$artifact_root/targets/targets_train.pt" \
    --validation-targets "$artifact_root/targets/targets_validation.pt" --output-dir "$probe_root" \
    --epochs "${EPOCHS:-30}" --batch-size "${BATCH_SIZE:-1}" --num-workers "${NUM_WORKERS:-2}" \
    --hidden-dim "${HIDDEN_DIM:-256}" --num-heads "${NUM_HEADS:-8}" --num-slots "${NUM_SLOTS:-6}" \
    --slot-depth "${SLOT_DEPTH:-1}" --transition-depth "${TRANSITION_DEPTH:-1}" \
    --interaction-depth "${INTERACTION_DEPTH:-1}" \
    --ffn-dim "${FFN_DIM:-1024}" --dropout "${DROPOUT:-0.1}" --learning-rate "${LR:-2e-4}" \
    --weight-decay "${WEIGHT_DECAY:-0.04}" --seed "${SEED:-239}" "${limit_args[@]}"
  if [[ "$probe_log_root" != "$probe_root" ]]; then
    local name
    for name in best_metrics.json history.json manifest.json; do
      [[ -f "$probe_root/$name" ]] && cp "$probe_root/$name" "$probe_log_root/$name"
    done
  fi
}
if [[ "$dry_run" == true ]]; then print_paths; printf 'epochs=%s\nnum_gpus=%s\n' "${EPOCHS:-30}" "${NUM_GPUS:-8}"; exit 0; fi
for path in "$artifact_root/targets/targets_train.pt" "$artifact_root/targets/targets_validation.pt" "$artifact_root/latents_nonpredictive/train/manifest.json" "$artifact_root/latents_nonpredictive/validation/manifest.json" "$artifact_root/latents_predictive/train/manifest.json" "$artifact_root/latents_predictive/validation/manifest.json"; do [[ -f "$path" ]] || { echo "missing probe input: $path" >&2; exit 1; }; done
if [[ "$internal_run" == true ]]; then run_stage; exit $?; fi
[[ ! -e "$probe_root" ]] || { echo "probe output exists; set PROBE_RUN_NAME to a new run name" >&2; exit 1; }
mkdir -p "$probe_log_root"
log=$probe_log_root/train_structured_probe.log
nohup bash "$0" --run --output-mode "$output_mode" >"$log" 2>&1 &
printf '%s\n' "$!" >"$probe_log_root/train_structured_probe.pid"
echo "started structured probe training: pid=$! log=$log"
