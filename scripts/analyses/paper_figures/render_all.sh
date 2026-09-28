#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/analyses/paper_figures/config.yaml
TASK=render_all

source tools/launcher/launch_from_config.sh "$@"
