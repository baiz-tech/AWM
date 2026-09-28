#!/usr/bin/env bash
set -euo pipefail
analysis_name=${ANALYSIS_NAME:-vjepa2_naive_probe_v5_decoder_validation_seed239_v1}
source "$(cd "$(dirname "$0")" && pwd)/analysis_common.sh"
parse_analysis_args "$@"

run_stage() {
  mkdir -p "$output_dir" "$log_dir"
  cd "$repo_root"
  args=()
  [[ -n ${MAX_SCENES:-} ]] && args+=(--max-scenes "$MAX_SCENES")
  CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} "$python_bin" "$script_dir/evaluate_structured_probe.py" \
    --cache-root "$cache_root" --targets "$targets" --checkpoint "$checkpoint" \
    --output-dir "$output_dir" --device cuda:0 --batch-size "${BATCH_SIZE:-4}" "${args[@]}"
  if [[ "$log_dir" != "$output_dir" ]]; then
    cp "$output_dir/metrics.json" "$log_dir/metrics.json"
  fi
}
if [[ "$dry_run" == true ]]; then print_analysis_paths; exit 0; fi
require_analysis_inputs
if [[ "$internal_run" == true ]]; then run_stage; exit $?; fi
[[ ! -e "$output_dir" && ! -e "$log_dir" ]] || { echo "refusing to overwrite analysis output" >&2; exit 1; }
mkdir -p "$log_dir"
nohup bash "$0" --run --output-mode "$output_mode" >"$log_dir/evaluate.log" 2>&1 &
printf '%s\n' "$!" >"$log_dir/evaluate.pid"
echo "started structured probe validation: pid=$! log=$log_dir/evaluate.log"
