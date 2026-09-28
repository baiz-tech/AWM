#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
OUTPUT_MODE=both; dry_run=0
while [[ $# -gt 0 ]]; do case "$1" in --output-mode) OUTPUT_MODE=$2; shift 2;; --dry-run) dry_run=1; shift;; *) echo "unknown argument: $1" >&2; exit 2;; esac; done
export OUTPUT_MODE
source "$script_dir/common.sh"
cmd=(bash "$0" --output-mode "$OUTPUT_MODE")
if [[ $dry_run == 1 ]]; then
  printf 'output_mode=%s\nartifact_run=%s\nlog_run=%s\nconfig=%s\npredictor=%s\ndataset=%s\nmaster_addr=%s\nmaster_port=%s\ncommand=' "$OUTPUT_MODE" "$artifact_run" "$log_run" "$config" "$predictor" "$dataset" "$master_addr" "$master_port"; printf '%q ' "${cmd[@]}"; echo; exit 0
fi
if [[ -e "$artifact_run" || -e "$log_run" ]]; then
  echo "output run exists; choose a new RUN_NAME or archive it first: $artifact_run" >&2
  exit 2
fi
mkdir -p "$log_run"
printf '%q ' "${cmd[@]}" >"$log_run/command.txt"; echo >>"$log_run/command.txt"
printf '{"output_mode":"%s","artifact_run":"%s","log_run":"%s","config":"%s","predictor":"%s","dataset":"%s","master_addr":"%s","master_port":%s}\n' "$OUTPUT_MODE" "$artifact_run" "$log_run" "$config" "$predictor" "$dataset" "$master_addr" "$master_port" >"$log_run/launcher_manifest.json"
nohup bash -c "source '$script_dir/common.sh'; \
  '$python_bin' -m recipe.$experiment.scripts.physionpp.prepare_targets --dataset-root '$dataset' --split data_v1 --output '$artifact_run/targets_data_v1.pt' --skip-missing-annotations && \
  '$python_bin' -m recipe.$experiment.scripts.physionpp.prepare_targets --dataset-root '$dataset' --split readout_data_v1 --output '$artifact_run/targets_readout_data_v1.pt' --skip-missing-annotations && \
  '$python_bin' -m recipe.$experiment.scripts.physionpp.cache_full_latents --config '$config' --predictor-checkpoint '$predictor' --split data_v1 --output-dir '$artifact_run/cache' --resume && \
  '$python_bin' -m recipe.$experiment.scripts.physionpp.cache_full_latents --config '$config' --predictor-checkpoint '$predictor' --split readout_data_v1 --output-dir '$artifact_run/cache' --resume && \
  torchrun --nnodes=1 --node_rank=0 --master_addr '$master_addr' --master_port '$master_port' --nproc_per_node=8 -m recipe.$experiment.scripts.physionpp.train_structured_probe --cache-root '$artifact_run/cache' --train-targets '$artifact_run/targets_data_v1.pt' --validation-targets '$artifact_run/targets_readout_data_v1.pt' --output-dir '$artifact_run/probe'" >"$log_run/pipeline.log" 2>&1 &
echo $! >"$log_run/pipeline.pid"
printf 'started pid=%s log=%s\n' "$!" "$log_run/pipeline.log"
