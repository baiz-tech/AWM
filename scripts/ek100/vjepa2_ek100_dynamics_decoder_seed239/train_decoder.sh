#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/ek100/vjepa2_ek100_dynamics_decoder_seed239/config.yaml
TASK=train_decoder

source tools/launcher/launch_from_config.sh "$@"
