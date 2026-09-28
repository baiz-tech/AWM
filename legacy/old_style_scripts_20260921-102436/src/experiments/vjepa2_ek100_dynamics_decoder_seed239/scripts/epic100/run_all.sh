#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

output_mode=both
dry_run=false
resume=false
internal_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-mode) output_mode=$2; shift 2 ;;
    --output-mode=*) output_mode=${1#*=}; shift ;;
    --dry-run) dry_run=true; shift ;;
    --resume) resume=true; shift ;;
    --run) internal_run=true; shift ;;
    *) echo "unsupported argument: $1" >&2; exit 2 ;;
  esac
done
resolve_output_paths

target_limit_args=()
if [[ -n "${MAX_EVENTS:-}" ]]; then target_limit_args+=(--max-events "$MAX_EVENTS"); fi
cache_resume_args=()
if [[ "$resume" == true ]]; then cache_resume_args+=(--resume); fi
train_resume_args=()
if [[ "$resume" == true && -f "$checkpoint_dir/latest.pt" ]]; then train_resume_args+=(--resume); fi

commands=(
  "$python_bin -m recipe.$experiment.scripts.epic100.prepare_targets --processed-root $processed_root --visor-root $visor_root --annotations $train_annotations --train-annotations $train_annotations --split train --output $train_targets ${target_limit_args[*]}"
  "$python_bin -m recipe.$experiment.scripts.epic100.prepare_targets --processed-root $processed_root --visor-root $visor_root --annotations $validation_annotations --train-annotations $train_annotations --split validation --output $validation_targets ${target_limit_args[*]}"
  "torchrun --nnodes=1 --node_rank=0 --master_addr $master_addr --master_port $master_port --nproc_per_node=8 -m recipe.$experiment.scripts.epic100.cache_full_latents --config $config --targets $train_targets --output-dir $train_cache ${cache_resume_args[*]}"
  "torchrun --nnodes=1 --node_rank=0 --master_addr $master_addr --master_port $((master_port + 1)) --nproc_per_node=8 -m recipe.$experiment.scripts.epic100.cache_full_latents --config $config --targets $validation_targets --output-dir $validation_cache ${cache_resume_args[*]}"
  "torchrun --nnodes=1 --node_rank=0 --master_addr $master_addr --master_port $((master_port + 2)) --nproc_per_node=8 -m recipe.$experiment.scripts.epic100.train_decoder --config $config --train-cache $train_cache --validation-cache $validation_cache --train-targets $train_targets --validation-targets $validation_targets --output-dir $checkpoint_dir ${train_resume_args[*]}"
  "$python_bin -m recipe.$experiment.scripts.epic100.evaluate_decoder --cache $validation_cache --targets $validation_targets --checkpoint $checkpoint_dir/best.pt --output $log_run/evaluations/validation/metrics.json"
)

if [[ "$dry_run" == true ]]; then
  printf 'output_mode=%s\nartifact_run=%s\nlog_run=%s\nconfig=%s\n' "$output_mode" "$artifact_run" "$log_run" "$config"
  printf 'processed_root=%s\nvisor_root=%s\ntrain_annotations=%s\nvalidation_annotations=%s\n' "$processed_root" "$visor_root" "$train_annotations" "$validation_annotations"
  printf 'official_checkpoint=%s\nmaster_addr=%s\nmaster_port=%s\n' "/data/ABDUCTIVE-WORLD/pretrain/checkpoints/vith.pt" "$master_addr" "$master_port"
  for index in "${!commands[@]}"; do printf 'command[%s]=%s\n' "$index" "${commands[$index]}"; done
  exit 0
fi

run_pipeline() {
  cd "$repo_root"
  source "$conda_sh"
  conda activate "$conda_env"
  mkdir -p "$artifact_run" "$log_run"
  if [[ "$resume" != true || ! -f "$train_targets" ]]; then eval "${commands[0]}"; fi
  if [[ "$resume" != true || ! -f "$validation_targets" ]]; then eval "${commands[1]}"; fi
  eval "${commands[2]}"
  eval "${commands[3]}"
  eval "${commands[4]}"
  eval "${commands[5]}"
}

if [[ "$internal_run" == true ]]; then run_pipeline; exit $?; fi
if [[ -e "$artifact_run" || ( "$log_run" != "$artifact_run" && -e "$log_run" ) ]]; then
  if [[ "$resume" != true ]]; then
    echo "output run exists; choose a new RUN_NAME or pass --resume: $artifact_run" >&2
    exit 2
  fi
fi
mkdir -p "$log_run"
launch=(bash "$0" --run --output-mode "$output_mode")
if [[ "$resume" == true ]]; then launch+=(--resume); fi
printf '%q ' "${launch[@]}" >"$log_run/command.txt"; echo >>"$log_run/command.txt"
cp "$config" "$log_run/config.resolved.yaml"
git -C "$repo_root" rev-parse HEAD >"$log_run/git_commit.txt"
git -C "$repo_root" status --short >"$log_run/git_status.txt"
printf '{"python":"%s","cuda_visible_devices":"%s","seed":239}\n' "$python_bin" "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" >"$log_run/environment.json"
printf '{"output_mode":"%s","artifact_run":"%s","log_run":"%s","config":"%s","official_checkpoint":"%s","processed_root":"%s","visor_root":"%s","master_addr":"%s","master_port":%s}\n' \
  "$output_mode" "$artifact_run" "$log_run" "$config" "/data/ABDUCTIVE-WORLD/pretrain/checkpoints/vith.pt" "$processed_root" "$visor_root" "$master_addr" "$master_port" >"$log_run/launcher_manifest.json"
nohup "${launch[@]}" >"$log_run/pipeline.log" 2>&1 &
printf '%s\n' "$!" >"$log_run/pipeline.pid"
echo "started pid=$! log=$log_run/pipeline.log"
