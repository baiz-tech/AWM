#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/ek100/awm_ek100_fair_attribution_seed239/config.yaml
TASK=train_e0

source tools/launcher/launch_from_config.sh "$@"
