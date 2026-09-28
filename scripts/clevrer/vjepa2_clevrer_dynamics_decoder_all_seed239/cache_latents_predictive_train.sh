#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/clevrer/vjepa2_clevrer_dynamics_decoder_all_seed239/config.yaml
TASK=cache_latents_predictive_train

source tools/launcher/launch_from_config.sh "$@"
