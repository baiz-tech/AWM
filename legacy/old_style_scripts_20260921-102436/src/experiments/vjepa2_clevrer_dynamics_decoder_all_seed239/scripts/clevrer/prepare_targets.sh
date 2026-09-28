#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
parse_common_args "$@"

train_target=$artifact_root/targets_train.pt
validation_target=$artifact_root/targets_validation.pt
run_stage() {
  mkdir -p "$artifact_root" "$artifact_log_root"
  cd "$repo_root"
  if [[ "$resume" == true && -f "$train_target" && -f "$validation_target" ]]; then
    echo "targets already complete; resume skips generation"
    return
  fi
  local max_args=()
  [[ -n ${MAX_SCENES:-} ]] && max_args+=(--max-scenes "$MAX_SCENES")
  local window_args=(--window-starts "${WINDOW_STARTS:-0}" --mode "${DECODER_MODE:-predictive}")
  "$python_bin" "$script_dir/prepare_targets.py" --split train --output "$train_target" "${window_args[@]}" "${max_args[@]}"
  "$python_bin" "$script_dir/prepare_targets.py" --split validation --output "$validation_target" "${window_args[@]}" "${max_args[@]}"
}
if [[ "$dry_run" == true ]]; then print_paths; printf 'train_target=%s\nvalidation_target=%s\n' "$train_target" "$validation_target"; exit 0; fi
if [[ "$internal_run" == true ]]; then run_stage; exit $?; fi
[[ ! -e "$train_target" && ! -e "$validation_target" ]] || { echo "targets exist; use --resume or a new output root" >&2; exit 1; }
mkdir -p "$artifact_log_root"
log=$artifact_log_root/prepare_targets.log
extra=(); [[ "$resume" == true ]] && extra+=(--resume)
nohup bash "$0" --run --output-mode "$output_mode" "${extra[@]}" >"$log" 2>&1 &
printf '%s\n' "$!" >"$artifact_log_root/prepare_targets.pid"
echo "started target generation: pid=$! log=$log"
