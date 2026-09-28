#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/physionpp/orca_physionpp_oracle_ocp_seed239/config.yaml
TASK=train_ocp

source tools/launcher/launch_from_config.sh "$@"
