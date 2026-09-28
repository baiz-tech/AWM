#!/usr/bin/env bash

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
python_bin=${PYTHON_BIN:-/data/shared/shared/envs/vjepa2/bin/python}
config=${CONFIG:-src/experiments/vjepa2_clevrer_dynamics_decoder_all_seed239/configs/clevrer-fullpatch-structured-probe-v1-8gpu.yaml}
world_checkpoint=${WORLD_CHECKPOINT:-/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/clevrer_vith_16to16_stride2_8gpu/best.pt}
data_experiment_root=${DATA_EXPERIMENT_ROOT:-/data/shyang/outputs/vjepa2-baiz/vjepa2_naive_probe_v5_decoder_clevrerAll}
workspace_experiment_root=${WORKSPACE_EXPERIMENT_ROOT:-$repo_root/outputs/runs/vjepa2_naive_probe_v5_decoder_clevrerAll}
probe_run_name=${PROBE_RUN_NAME:-clevrer_dynamics_decoder_seed239_v1}

parse_common_args() {
  output_mode=${OUTPUT_MODE:-both}
  dry_run=false
  internal_run=false
  resume=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output-mode)
        [[ $# -ge 2 ]] || { echo "--output-mode requires a value" >&2; return 2; }
        output_mode=$2
        shift 2
        ;;
      --output-mode=*) output_mode=${1#*=}; shift ;;
      --dry-run) dry_run=true; shift ;;
      --run) internal_run=true; shift ;;
      --resume) resume=true; shift ;;
      *) echo "unsupported argument: $1" >&2; return 2 ;;
    esac
  done
  case "$output_mode" in
    both|data|workspace) ;;
    *) echo "unsupported --output-mode: $output_mode" >&2; return 2 ;;
  esac
  if [[ "$output_mode" == workspace ]]; then
    primary_experiment_root=$workspace_experiment_root
  else
    primary_experiment_root=$data_experiment_root
  fi
  if [[ "$output_mode" == both ]]; then
    log_experiment_root=$workspace_experiment_root
  else
    log_experiment_root=$primary_experiment_root
  fi
  artifact_root=$primary_experiment_root/clevrer_fullpatch_data_v1
  probe_root=$primary_experiment_root/$probe_run_name
  artifact_log_root=$log_experiment_root/clevrer_fullpatch_data_v1
  probe_log_root=$log_experiment_root/$probe_run_name
}

print_paths() {
  printf 'output_mode=%s\n' "$output_mode"
  printf 'primary_experiment_root=%s\n' "$primary_experiment_root"
  printf 'log_experiment_root=%s\n' "$log_experiment_root"
  printf 'artifact_root=%s\n' "$artifact_root"
  printf 'probe_run_name=%s\n' "$probe_run_name"
  printf 'probe_root=%s\n' "$probe_root"
  printf 'world_checkpoint=%s\n' "$world_checkpoint"
  printf 'window_starts=%s\n' "${WINDOW_STARTS:-0}"
  printf 'decoder_mode=%s\n' "${DECODER_MODE:-predictive}"
}
