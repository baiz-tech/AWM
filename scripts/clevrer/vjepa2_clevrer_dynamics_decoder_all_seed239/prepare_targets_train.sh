#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/clevrer/vjepa2_clevrer_dynamics_decoder_all_seed239/config.yaml
TASK=prepare_targets_train

source tools/launcher/launch_from_config.sh "$@"
