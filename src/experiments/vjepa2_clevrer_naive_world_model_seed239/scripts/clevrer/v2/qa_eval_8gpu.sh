#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-src/experiments/vjepa2_clevrer_naive_world_model_seed239/configs/clevrer/clevrer-vith-16to16-v2-8gpu.yaml}
TASK=${TASK:-qa_eval}
VARIANT=current_future
export OUTPUT_MODE=${OUTPUT_MODE:-workspace}
export WORKSPACE_MIRROR_MODE=${WORKSPACE_MIRROR_MODE:-lightweight}
if [[ " ${*} " != *" --output-mode "* && " ${*} " != *" --output-mode="* ]]; then
  set -- --output-mode "$OUTPUT_MODE" "$@"
fi
source tools/launcher/launch_from_config.sh
