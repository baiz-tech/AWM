#!/usr/bin/env bash

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
python_bin=${PYTHON_BIN:-/data/shared/envs/vjepa2-312/bin/python}
data_experiment_root=${DATA_EXPERIMENT_ROOT:-/data/shyang/outputs/vjepa2-baiz/vjepa2_naive_probe_v5_decoder}
workspace_experiment_root=${WORKSPACE_EXPERIMENT_ROOT:-$repo_root/outputs/runs/vjepa2_naive_probe_v5_decoder}
data_analysis_root=${DATA_ANALYSIS_ROOT:-/data/shyang/outputs/vjepa2-baiz/analyses}
workspace_analysis_root=${WORKSPACE_ANALYSIS_ROOT:-$repo_root/outputs/analyses}
probe_run_name=${PROBE_RUN_NAME:-clevrer_dynamics_decoder_seed239_v1}

parse_analysis_args() {
  output_mode=${OUTPUT_MODE:-both}
  dry_run=false
  internal_run=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output-mode) output_mode=$2; shift 2 ;;
      --output-mode=*) output_mode=${1#*=}; shift ;;
      --dry-run) dry_run=true; shift ;;
      --run) internal_run=true; shift ;;
      *) echo "unsupported argument: $1" >&2; return 2 ;;
    esac
  done
  case "$output_mode" in
    workspace)
      input_root=$workspace_experiment_root
      output_dir=${OUTPUT_DIR:-$workspace_analysis_root/$analysis_name}
      log_dir=$output_dir
      ;;
    data)
      input_root=$data_experiment_root
      output_dir=${OUTPUT_DIR:-$data_analysis_root/$analysis_name}
      log_dir=$output_dir
      ;;
    both)
      input_root=$data_experiment_root
      output_dir=${OUTPUT_DIR:-$data_analysis_root/$analysis_name}
      log_dir=$workspace_analysis_root/$analysis_name
      ;;
    *) echo "unsupported --output-mode: $output_mode" >&2; return 2 ;;
  esac
  cache_root=${CACHE_ROOT:-$input_root/clevrer_fullpatch_data_v1/latents}
  targets=${TARGETS:-$input_root/clevrer_fullpatch_data_v1/targets_validation.pt}
  checkpoint=${PROBE_CHECKPOINT:-$input_root/$probe_run_name/best.pt}
}

print_analysis_paths() {
  printf 'output_mode=%s\n' "$output_mode"
  printf 'cache_root=%s\n' "$cache_root"
  printf 'targets=%s\n' "$targets"
  printf 'checkpoint=%s\n' "$checkpoint"
  printf 'output_dir=%s\n' "$output_dir"
  printf 'log_dir=%s\n' "$log_dir"
}

require_analysis_inputs() {
  [[ -d "$cache_root/validation" ]] || { echo "missing validation cache: $cache_root/validation" >&2; return 1; }
  [[ -f "$targets" ]] || { echo "missing validation targets: $targets" >&2; return 1; }
  [[ -f "$checkpoint" ]] || { echo "missing probe checkpoint: $checkpoint" >&2; return 1; }
}
