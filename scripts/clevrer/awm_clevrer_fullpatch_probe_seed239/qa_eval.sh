#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/clevrer/awm_clevrer_fullpatch_probe_seed239/config.yaml}
TASK=qa_eval

source tools/launcher/launch_from_config.sh "$@"
