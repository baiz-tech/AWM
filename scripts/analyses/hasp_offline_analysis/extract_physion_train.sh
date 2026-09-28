#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/analyses/hasp_offline_analysis/config.yaml
TASK=extract_physion_train

source tools/launcher/launch_from_config.sh "$@"
