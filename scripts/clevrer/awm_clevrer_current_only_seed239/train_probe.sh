#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/clevrer/awm_clevrer_current_only_seed239/config.yaml
TASK=train_probe

source tools/launcher/launch_from_config.sh "$@"
