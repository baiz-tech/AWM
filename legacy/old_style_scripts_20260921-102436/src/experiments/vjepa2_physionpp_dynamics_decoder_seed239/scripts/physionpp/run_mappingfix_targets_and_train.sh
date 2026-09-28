#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
run_name=${RUN_NAME:-physionpp_segmentation_2d_decoder_seed239_v2}
probe_name=${PROBE_NAME:-probe_mappingfix_v1}
output_mode=${OUTPUT_MODE:-workspace}
num_gpus=${NUM_GPUS:-8}
master_port=${MASTER_PORT:-29643}
epochs=${EPOCHS:-20}
batch_size=${BATCH_SIZE:-2}
num_workers=${NUM_WORKERS:-2}
dataset_root=${DATASET_ROOT:-/data/ABDUCTIVE-WORLD/physion_v2/extracted}
python_bin=${PYTHON_BIN:-python}

export OUTPUT_MODE="$output_mode" RUN_NAME="$run_name"
source "$script_dir/common.sh"

cache_root="$artifact_run/cache"
train_targets="$artifact_run/targets_data_v2.pt"
validation_targets="$artifact_run/targets_readout_data_v2.pt"
log_file="$log_run/${probe_name}_pipeline.log"
pid_file="$log_run/${probe_name}_pipeline.pid"

for path in "$cache_root/data_v1" "$cache_root/readout_data_v1"; do
    [[ -d "$path" ]] || { echo "missing latent cache: $path" >&2; exit 1; }
done
[[ ! -e "$artifact_run/$probe_name" ]] || { echo "output already exists: $artifact_run/$probe_name" >&2; exit 2; }
mkdir -p "$log_run"

command_file="$log_run/${probe_name}_pipeline_command.txt"
cat > "$command_file" <<EOF
run_name=$run_name
probe_name=$probe_name
output_mode=$output_mode
dataset_root=$dataset_root
cache_root=$cache_root
train_targets=$train_targets
validation_targets=$validation_targets
num_gpus=$num_gpus
master_addr=$master_addr
master_port=$master_port
epochs=$epochs
batch_size=$batch_size
num_workers=$num_workers
EOF

nohup bash -c "
set -euo pipefail
DATASET_ROOT='$dataset_root' SPLIT=data_v1 OUTPUT='$train_targets' WORK_DIR='$artifact_run/targets_data_v2.shards' NUM_GPUS='$num_gpus' PYTHON_BIN='$python_bin' bash '$script_dir/prepare_targets_8gpu.sh'
DATASET_ROOT='$dataset_root' SPLIT=readout_data_v1 OUTPUT='$validation_targets' WORK_DIR='$artifact_run/targets_readout_data_v2.shards' NUM_GPUS='$num_gpus' PYTHON_BIN='$python_bin' bash '$script_dir/prepare_targets_8gpu.sh'
RUN_NAME='$run_name' OUTPUT_MODE='$output_mode' PROBE_NAME='$probe_name' CACHE_ROOT='$cache_root' TRAIN_TARGETS='$train_targets' VALIDATION_TARGETS='$validation_targets' MASTER_PORT='$master_port' EPOCHS='$epochs' BATCH_SIZE='$batch_size' NUM_WORKERS='$num_workers' bash '$script_dir/train_decoder_only.sh'
" > "$log_file" 2>&1 &
echo $! > "$pid_file"
printf 'started pid=%s\nlog=%s\nprobe=%s/%s\n' "$!" "$log_file" "$artifact_run" "$probe_name"
