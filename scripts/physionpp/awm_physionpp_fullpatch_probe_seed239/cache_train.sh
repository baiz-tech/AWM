#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/physionpp/awm_physionpp_fullpatch_probe_seed239/config.yaml}
TASK=cache_train

source tools/launcher/launch_from_config.sh "$@"
