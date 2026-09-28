#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/ek100/awm_ek100_multiscale_adapter_seed239/config.yaml
TASK=train_readout

source tools/launcher/launch_from_config.sh "$@"
