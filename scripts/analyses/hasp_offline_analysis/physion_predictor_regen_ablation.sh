#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/analyses/hasp_offline_analysis/config.yaml
TASK=physion_predictor_regen_ablation

source tools/launcher/launch_from_config.sh "$@"
