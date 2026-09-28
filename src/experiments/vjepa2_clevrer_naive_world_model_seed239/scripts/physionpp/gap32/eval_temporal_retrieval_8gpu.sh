#!/usr/bin/env bash
set -euo pipefail

CONFIG=src/experiments/vjepa2_clevrer_naive_world_model_seed239/configs/physionpp/physion-vith-16to16-gap32-8gpu.yaml
TASK=same_video_temporal_retrieval
export OUTPUT_MODE=${OUTPUT_MODE:-workspace}
export WORKSPACE_MIRROR_MODE=${WORKSPACE_MIRROR_MODE:-lightweight}
source tools/launcher/launch_from_config.sh
