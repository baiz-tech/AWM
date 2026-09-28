#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/analyses/clevrer_intervention/config.yaml
TASK=export_predictive_latents

source tools/launcher/launch_from_config.sh "$@"
