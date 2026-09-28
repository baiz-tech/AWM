#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/physionpp/videomae2_physionpp_oracle_ocp_seed239/config.yaml
TASK=cache_train

source tools/launcher/launch_from_config.sh "$@"
