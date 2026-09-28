#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/analyses/hasp_offline_analysis/config.yaml
TASK=physion_relation_instance_ablation

source tools/launcher/launch_from_config.sh "$@"
