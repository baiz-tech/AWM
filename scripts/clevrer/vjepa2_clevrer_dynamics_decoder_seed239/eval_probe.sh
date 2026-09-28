#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/clevrer/vjepa2_clevrer_dynamics_decoder_seed239/config.yaml
TASK=eval_probe

source tools/launcher/launch_from_config.sh "$@"
