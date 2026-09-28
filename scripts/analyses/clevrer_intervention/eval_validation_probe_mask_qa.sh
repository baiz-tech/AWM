#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/analyses/clevrer_intervention/config.yaml
TASK=eval_validation_probe_mask_qa

source tools/launcher/launch_from_config.sh "$@"
