#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/ek100/videomae2_ek100_finetune_seed239/config.yaml
TASK=prepare_indices_train

source tools/launcher/launch_from_config.sh "$@"
