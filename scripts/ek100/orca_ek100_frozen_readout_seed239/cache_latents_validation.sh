#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/ek100/orca_ek100_frozen_readout_seed239/config.yaml
TASK=cache_latents_validation

source tools/launcher/launch_from_config.sh "$@"
