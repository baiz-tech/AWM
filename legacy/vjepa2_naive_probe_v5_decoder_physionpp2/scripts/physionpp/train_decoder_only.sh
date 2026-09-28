#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output_mode=${OUTPUT_MODE:-workspace}
run_name=${RUN_NAME:-physionpp_segmentation_2d_decoder_seed239_v2}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29641}
epochs=${EPOCHS:-20}
batch_size=${BATCH_SIZE:-2}
num_workers=${NUM_WORKERS:-2}
probe_name=${PROBE_NAME:-probe}
cache_override=${CACHE_ROOT:-}
train_targets_override=${TRAIN_TARGETS:-}
validation_targets_override=${VALIDATION_TARGETS:-}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-mode) output_mode=$2; shift 2 ;;
        --run-name) run_name=$2; shift 2 ;;
        --master-addr) master_addr=$2; shift 2 ;;
        --master-port) master_port=$2; shift 2 ;;
        --epochs) epochs=$2; shift 2 ;;
        --batch-size) batch_size=$2; shift 2 ;;
        --num-workers) num_workers=$2; shift 2 ;;
        --probe-name) probe_name=$2; shift 2 ;;
        --cache-root) cache_override=$2; shift 2 ;;
        --train-targets) train_targets_override=$2; shift 2 ;;
        --validation-targets) validation_targets_override=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

export OUTPUT_MODE="$output_mode"
export RUN_NAME="$run_name"
source "$script_dir/common.sh"

target_train=${train_targets_override:-$artifact_run/targets_data_v1.pt}
target_validation=${validation_targets_override:-$artifact_run/targets_readout_data_v1.pt}
cache_root=${cache_override:-$artifact_run/cache}
probe_dir="$artifact_run/$probe_name"
log_file="$log_run/${probe_name}_train.log"
pid_file="$log_run/${probe_name}_train.pid"

for path in "$target_train" "$target_validation" "$cache_root/data_v1" "$cache_root/readout_data_v1"; do
    if [[ ! -e "$path" ]]; then
        echo "missing decoder input: $path" >&2
        exit 1
    fi
done

if [[ -e "$probe_dir" ]]; then
    echo "probe output already exists; choose another RUN_NAME or remove/archive it: $probe_dir" >&2
    exit 2
fi

mkdir -p "$log_run"
command_file="$log_run/${probe_name}_train_command.txt"
cat > "$command_file" <<EOF
output_mode=$output_mode
artifact_run=$artifact_run
log_run=$log_run
master_addr=$master_addr
master_port=$master_port
epochs=$epochs
batch_size=$batch_size
num_workers=$num_workers
probe_name=$probe_name
cache_root=$cache_root
train_targets=$target_train
validation_targets=$target_validation
EOF

nohup "$python_bin" -m torch.distributed.run \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr "$master_addr" \
    --master_port "$master_port" \
    --nproc_per_node=8 \
    -m recipe."$experiment".scripts.physionpp.train_structured_probe \
    --cache-root "$cache_root" \
    --train-targets "$target_train" \
    --validation-targets "$target_validation" \
    --output-dir "$probe_dir" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --num-workers "$num_workers" \
    > "$log_file" 2>&1 &

echo $! > "$pid_file"
printf 'started pid=%s\nlog=%s\nprobe=%s\n' "$!" "$log_file" "$probe_dir"
