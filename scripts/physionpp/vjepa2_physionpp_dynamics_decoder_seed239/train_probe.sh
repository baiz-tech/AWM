#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/physionpp/vjepa2_physionpp_dynamics_decoder_seed239/config.yaml
TASK=train_probe

source tools/launcher/launch_from_config.sh "$@"
