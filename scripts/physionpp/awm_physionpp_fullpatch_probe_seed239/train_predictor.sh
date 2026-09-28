#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/physionpp/awm_physionpp_fullpatch_probe_seed239/config.yaml
TASK=train_predictor

source tools/launcher/launch_from_config.sh "$@"
