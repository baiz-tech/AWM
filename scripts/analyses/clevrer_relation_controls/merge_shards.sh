#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/analyses/clevrer_relation_controls/config.yaml
TASK=merge_shards

source tools/launcher/launch_from_config.sh "$@"
