#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/clevrer/awm_clevrer_fullpatch_probe_seed239/config.yaml
TASK=train_probe

source tools/launcher/launch_from_config.sh "$@"
