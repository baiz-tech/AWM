#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
parse_common_args "$@"

run_stage() {
  cd "$repo_root"
  bash "$script_dir/prepare_targets.sh" --run --resume --output-mode "$output_mode"
  # The probe trainer consumes both controlled (real-future) and predictive
  # (rollout-future) caches. Keep the two modes in separate directories.
  DECODER_MODE=nonpredictive LATENT_CACHE_NAME=latents_nonpredictive \
    bash "$script_dir/cache_full_latents.sh" --run --resume --output-mode "$output_mode"
  DECODER_MODE=predictive LATENT_CACHE_NAME=latents_predictive \
    bash "$script_dir/cache_full_latents.sh" --run --resume --output-mode "$output_mode"
  bash "$script_dir/train_structured_probe_8gpu.sh" --run --output-mode "$output_mode"
}
if [[ "$dry_run" == true ]]; then
  print_paths
  bash "$script_dir/prepare_targets.sh" --dry-run --output-mode "$output_mode"
  DECODER_MODE=nonpredictive LATENT_CACHE_NAME=latents_nonpredictive \
    bash "$script_dir/cache_full_latents.sh" --dry-run --output-mode "$output_mode"
  DECODER_MODE=predictive LATENT_CACHE_NAME=latents_predictive \
    bash "$script_dir/cache_full_latents.sh" --dry-run --output-mode "$output_mode"
  bash "$script_dir/train_structured_probe_8gpu.sh" --dry-run --output-mode "$output_mode"
  exit 0
fi
if [[ "$internal_run" == true ]]; then run_stage; exit $?; fi
mkdir -p "$artifact_log_root"
log=$artifact_log_root/run_all.log
nohup bash "$0" --run --output-mode "$output_mode" >"$log" 2>&1 &
printf '%s\n' "$!" >"$artifact_log_root/run_all.pid"
echo "started targets -> full cache -> structured probe: pid=$! log=$log"
