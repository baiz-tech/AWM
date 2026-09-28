#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/ek100/awm_ek100_multiscale_adapter_seed239/config.yaml}
TASK=prepare_indices_train

source tools/launcher/launch_from_config.sh "$@"
