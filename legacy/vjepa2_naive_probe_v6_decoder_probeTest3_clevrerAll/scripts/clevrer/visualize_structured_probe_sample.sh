#!/usr/bin/env bash
set -euo pipefail
analysis_name=${ANALYSIS_NAME:-vjepa2_naive_probe_v6_decoder_probeTest3_predictive_sample_visualization_seed239}
source "$(cd "$(dirname "$0")" && pwd)/analysis_common.sh"
parse_analysis_args "$@"

run_stage() {
  mkdir -p "$output_dir" "$log_dir"
  cd "$repo_root"
  args=()
  [[ -n ${SCENE_ID:-} ]] && args+=(--scene-id "$SCENE_ID")
  [[ -n ${WINDOW_START:-} ]] && args+=(--window-start "$WINDOW_START")
  CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} "$python_bin" "$script_dir/inspect_structured_probe_scene.py" \
    --cache-root "$cache_root" --targets "$targets" --checkpoint "$checkpoint" \
    --output-dir "$output_dir" --device cuda:0 --seed "${SEED:-20260814}" \
    --render-video "${args[@]}"
  if [[ "$log_dir" != "$output_dir" ]]; then
    cp "$output_dir/scene_report.json" "$output_dir/scene_report.md" "$log_dir/"
  fi
}
if [[ "$dry_run" == true ]]; then print_analysis_paths; exit 0; fi
require_analysis_inputs
if [[ "$internal_run" == true ]]; then run_stage; exit $?; fi
[[ ! -e "$output_dir" && ! -e "$log_dir" ]] || { echo "refusing to overwrite analysis output" >&2; exit 1; }
mkdir -p "$log_dir"
nohup bash "$0" --run --output-mode "$output_mode" >"$log_dir/visualize.log" 2>&1 &
printf '%s\n' "$!" >"$log_dir/visualize.pid"
echo "started structured probe visualization: pid=$! log=$log_dir/visualize.log"
