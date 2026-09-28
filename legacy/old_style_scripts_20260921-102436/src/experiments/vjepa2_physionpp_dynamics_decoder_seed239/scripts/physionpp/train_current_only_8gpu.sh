#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
run_name=${RUN_NAME:-physionpp3_current_only_predictability_seed239_v1}
cache_root=${CACHE_ROOT:-$repo_root/outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp2/physionpp_segmentation_2d_decoder_seed239_v2/cache}
train_targets=${TRAIN_TARGETS:-$repo_root/outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/targets/targets_data_v1.pt}
validation_targets=${VALIDATION_TARGETS:-$repo_root/outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/targets/targets_readout_data_v1.pt}
output_dir=${OUTPUT_DIR:-$repo_root/outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/$run_name/current_only_probe}
log_file=${LOG_FILE:-$repo_root/outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/$run_name/train.log}
pid_file=${PID_FILE:-$repo_root/outputs/runs/vjepa2_naive_probe_v5_decoder_physionpp3/$run_name/train.pid}
epochs=${EPOCHS:-20}; batch_size=${BATCH_SIZE:-2}; num_workers=${NUM_WORKERS:-2}; master_port=${MASTER_PORT:-29654}; python_bin=${PYTHON_BIN:-python}
[[ -d "$cache_root/data_v1" && -d "$cache_root/readout_data_v1" ]] || { echo "latent cache 不完整: $cache_root" >&2; exit 1; }
[[ -f "$train_targets" && -f "$validation_targets" ]] || { echo "targets 不完整" >&2; exit 1; }
[[ ! -e "$output_dir" ]] || { echo "输出目录已存在，请更换 RUN_NAME: $output_dir" >&2; exit 2; }
mkdir -p "$(dirname "$log_file")"; cat > "$(dirname "$log_file")/command.txt" <<EOF
run_name=$run_name
cache_root=$cache_root
train_targets=$train_targets
validation_targets=$validation_targets
output_dir=$output_dir
epochs=$epochs
batch_size=$batch_size
num_workers=$num_workers
master_port=$master_port
EOF
nohup "$python_bin" -m torch.distributed.run --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$master_port" --nproc_per_node=8 -m src.experiments.vjepa2_physionpp_dynamics_decoder_seed239.scripts.physionpp.train_current_only --cache-root "$cache_root" --train-targets "$train_targets" --validation-targets "$validation_targets" --output-dir "$output_dir" --epochs "$epochs" --batch-size "$batch_size" --num-workers "$num_workers" > "$log_file" 2>&1 &
echo $! > "$pid_file"
printf 'started pid=%s\nlog=%s\noutput=%s\n' "$!" "$log_file" "$output_dir"
