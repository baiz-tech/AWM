#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/clevrer/vjepa2_clevrer_naive_world_model_seed239/config.yaml
TASK=qa_eval

source tools/launcher/launch_from_config.sh "$@"
