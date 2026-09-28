#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
python_bin=${PYTHON_BIN:-/data/shared/envs/vjepa2-312/bin/python}
workspace_root=${WORKSPACE_EXPERIMENT_ROOT:-$repo_root/outputs/runs/vjepa2_naive_probe_v6_decoder_probeTest2_clevrerAll}
analysis_root=${WORKSPACE_ANALYSIS_ROOT:-$repo_root/outputs/analyses}
data_root=${DATA_EXPERIMENT_ROOT:-/data/shyang/outputs/vjepa2-baiz/vjepa2_naive_probe_v6_decoder_probeTest2_clevrerAll}
probe_run_name=${PROBE_RUN_NAME:-clevrer_direct_zdyn_probe_seed239_v4}
dataset_root=${DATASET_ROOT:-/data/shared/datasets/CLEVRER}
seed=${SEED:-20260814}
output_mode=${OUTPUT_MODE:-workspace}
scene_id=${SCENE_ID:-}
window_start=${WINDOW_START:-}
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --output-mode) output_mode=${2:?--output-mode requires a value}; shift 2 ;;
    --output-mode=*) output_mode=${1#*=}; shift ;;
    *) echo "unsupported argument: $1" >&2; exit 2 ;;
  esac
done

case "$output_mode" in
  workspace) input_root="$workspace_root"; output_base="$analysis_root" ;;
  data|both) input_root="$data_root"; output_base="${DATA_ANALYSIS_ROOT:-/data/shyang/outputs/vjepa2-baiz/analyses}" ;;
  *) echo "unsupported --output-mode: $output_mode" >&2; exit 2 ;;
esac

artifact_root=${INPUT_ARTIFACT_ROOT:-$input_root/clevrer_fullpatch_data_v1}
targets=${TARGETS:-$artifact_root/targets/targets_validation.pt}
checkpoint=${PROBE_CHECKPOINT:-$input_root/$probe_run_name/best.pt}

# Test2 training runs may intentionally reuse Test1's targets/latent caches.
# When the local Test2 artifact directory is absent, recover the exact input
# locations recorded in the probe checkpoint manifest instead of guessing.
if [[ -z "${INPUT_ARTIFACT_ROOT:-}" && -z "${TARGETS:-}" && ! -f "$targets" && -f "$checkpoint" ]]; then
  checkpoint_manifest="$(dirname "$checkpoint")/manifest.json"
  manifest_artifact=$($python_bin -c 'import json,sys; p=json.load(open(sys.argv[1])); x=p.get("validation_targets"); print(__import__("pathlib").Path(x).resolve().parents[1] if x else "")' "$checkpoint_manifest" 2>/dev/null || true)
  if [[ -n "$manifest_artifact" && -f "$manifest_artifact/targets/targets_validation.pt" ]]; then
    artifact_root="$manifest_artifact"
    targets="$artifact_root/targets/targets_validation.pt"
  fi
fi
analysis_name=${ANALYSIS_NAME:-vjepa2_naive_probe_v6_decoder_probeTest2_sample_visualization_seed${seed}}
output_root=${OUTPUT_DIR:-$output_base/$analysis_name}
selection="$output_root/selection.json"

if [[ "$dry_run" == true ]]; then
  printf 'output_mode=%s\ninput_root=%s\nartifact_root=%s\ntargets=%s\ncheckpoint=%s\ndataset_root=%s\noutput_root=%s\nseed=%s\nscene_id=%s\nwindow_start=%s\n' \
    "$output_mode" "$input_root" "$artifact_root" "$targets" "$checkpoint" "$dataset_root" "$output_root" "$seed" "${scene_id:-random}" "${window_start:-first-for-scene}"
  exit 0
fi

[[ -f "$targets" ]] || { echo "missing validation targets: $targets" >&2; exit 1; }
[[ -f "$checkpoint" ]] || { echo "missing probe checkpoint: $checkpoint" >&2; exit 1; }
for mode in nonpredictive predictive; do
  [[ -d "$artifact_root/latents_$mode/validation" ]] || { echo "missing $mode cache" >&2; exit 1; }
done
[[ -d "$dataset_root/videos/val" ]] || { echo "missing validation videos: $dataset_root/videos/val" >&2; exit 1; }
[[ ! -e "$output_root" ]] || { echo "refusing to overwrite visualization output: $output_root" >&2; exit 1; }
mkdir -p "$output_root"

common_args=(--targets "$targets" --checkpoint "$checkpoint" --dataset-root "$dataset_root" --seed "$seed" --device cuda:0)
[[ -n "$scene_id" ]] && common_args+=(--scene-id "$scene_id")
[[ -n "$window_start" ]] && common_args+=(--window-start "$window_start")

for mode in nonpredictive predictive; do
  mode_output="$output_root/$mode"
  mkdir -p "$mode_output"
  mode_args=(--cache-root "$artifact_root/latents_$mode" --output-dir "$mode_output" --mode-label "$mode" "${common_args[@]}")
  if [[ "$mode" == nonpredictive ]]; then
    mode_args+=(--selection-output "$selection")
  else
    [[ -f "$selection" ]] || { echo "selection file missing after nonpredictive render: $selection" >&2; exit 1; }
    selected_scene=$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["scene_id"])' "$selection")
    selected_window=$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["window_start"])' "$selection")
    mode_args+=(--scene-id "$selected_scene" --window-start "$selected_window")
  fi
  "$python_bin" "$script_dir/inspect_structured_probe_scene.py" "${mode_args[@]}" --render-video
done

if [[ "$output_mode" == both ]]; then
  mkdir -p "$workspace_root/outputs/analyses/$analysis_name"
  cp -a "$output_root/." "$workspace_root/outputs/analyses/$analysis_name/"
fi
