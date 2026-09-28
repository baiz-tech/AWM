#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
source "$script_dir/common.sh"
output_mode=${OUTPUT_MODE:-workspace}
previous=""
for argument in "$@"; do
  if [[ "$previous" == --output-mode ]]; then output_mode=$argument; fi
  [[ "$argument" == --output-mode=* ]] && output_mode=${argument#*=}
  previous=$argument
done
case "$output_mode" in both|data) probe_base=$data_experiment_root ;; workspace) probe_base=$workspace_experiment_root ;; *) echo "invalid output mode: $output_mode" >&2; exit 2 ;; esac
export PROBE_CHECKPOINT=${PROBE_CHECKPOINT:-$probe_base/$probe_run_name/best.pt}
export CONFIG=${CONFIG:-src/experiments/vjepa2_clevrer_dynamics_decoder_all_seed239/configs/clevrer-fullpatch-structured-probe-v1-8gpu.yaml}
export TASK=${TASK:-qa_eval}
export VARIANT=current_future
if [[ " ${*} " != *" --output-mode "* && " ${*} " != *" --output-mode="* ]]; then
  set -- --output-mode "$output_mode" "$@"
fi
source tools/launcher/launch_from_config.sh
