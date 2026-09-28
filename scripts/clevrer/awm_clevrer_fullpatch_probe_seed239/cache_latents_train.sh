#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/clevrer/awm_clevrer_fullpatch_probe_seed239/config.yaml
TASK=cache_latents_train

source tools/launcher/launch_from_config.sh "$@"
