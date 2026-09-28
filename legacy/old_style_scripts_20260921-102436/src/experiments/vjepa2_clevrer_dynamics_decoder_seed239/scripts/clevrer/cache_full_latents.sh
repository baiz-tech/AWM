#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
parse_common_args "$@"

run_stage() {
  mkdir -p "$artifact_root" "$artifact_log_root"
  cd "$repo_root"
  local resume_args=()
  [[ "$resume" == true ]] && resume_args+=(--resume)
  local max_args=()
  [[ -n ${MAX_SCENES:-} ]] && max_args+=(--max-scenes "$MAX_SCENES")
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} "$python_bin" -m torch.distributed.run \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=${CACHE_PORT:-29710} \
    --nproc_per_node=${NUM_GPUS:-8} "$script_dir/cache_full_latents.py" \
    --config "$config" --split train --checkpoint "$world_checkpoint" --output-dir "$artifact_root/latents" \
    --window-starts "${WINDOW_STARTS:-0,32,64}" \
    --batch-size "${CACHE_BATCH_SIZE:-1}" --num-workers "${CACHE_NUM_WORKERS:-2}" "${resume_args[@]}" "${max_args[@]}"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} "$python_bin" -m torch.distributed.run \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=${CACHE_PORT_VALIDATION:-29711} \
    --nproc_per_node=${NUM_GPUS:-8} "$script_dir/cache_full_latents.py" \
    --config "$config" --split validation --checkpoint "$world_checkpoint" --output-dir "$artifact_root/latents" \
    --window-starts "${WINDOW_STARTS:-0,32,64}" \
    --batch-size "${CACHE_BATCH_SIZE:-1}" --num-workers "${CACHE_NUM_WORKERS:-2}" "${resume_args[@]}" "${max_args[@]}"
  if [[ "$artifact_log_root" != "$artifact_root" ]]; then
    mkdir -p "$artifact_log_root/latents/train" "$artifact_log_root/latents/validation"
    cp "$artifact_root/latents/train/manifest.json" "$artifact_log_root/latents/train/manifest.json"
    cp "$artifact_root/latents/validation/manifest.json" "$artifact_log_root/latents/validation/manifest.json"
  fi
}
if [[ "$dry_run" == true ]]; then print_paths; printf 'latent_root=%s\nnum_gpus=%s\n' "$artifact_root/latents" "${NUM_GPUS:-8}"; exit 0; fi
[[ -f "$world_checkpoint" ]] || { echo "missing world checkpoint: $world_checkpoint" >&2; exit 1; }
if [[ "$internal_run" == true ]]; then run_stage; exit $?; fi
[[ ! -e "$artifact_root/latents" || "$resume" == true ]] || { echo "latent cache exists; use --resume" >&2; exit 1; }
mkdir -p "$artifact_log_root"
log=$artifact_log_root/cache_full_latents.log
extra=(); [[ "$resume" == true ]] && extra+=(--resume)
nohup bash "$0" --run --output-mode "$output_mode" "${extra[@]}" >"$log" 2>&1 &
printf '%s\n' "$!" >"$artifact_log_root/cache_full_latents.pid"
echo "started full-patch cache export: pid=$! log=$log"
