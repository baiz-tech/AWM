#!/usr/bin/env bash
set -euo pipefail
repo=${REPO:-/data/shared/user/workspace/work/vjepa2-baiz}
config=${QA_CONFIG:-$repo/src/experiments/vjepa2_clevrer_dynamics_decoder_all_seed239/configs/clevrer-qa-decoder-all-32gpu.yaml}
scene_root=${QA_SCENE_ROOT:-/data/shared/user/outputs/vjepa2-baiz/vjepa2_naive_probe_v5_decoder_clevrerAll/clevrer_qa_decoder_all/scene_outputs}
output_dir=${QA_OUTPUT_DIR:-/data/shared/user/outputs/vjepa2-baiz/vjepa2_naive_probe_v5_decoder_clevrerAll/clevrer_qa_decoder_all/qa_checkpoints}
master_addr=${MASTER_ADDR:-192.0.2.10}; master_port=${MASTER_PORT:-29510}
hosts=(192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14); ifaces=(eth0 eth0 eth0 eth1)
smoke=false; dry_run=false
while [[ $# -gt 0 ]]; do case "$1" in --smoke-test) smoke=true; shift;; --dry-run) dry_run=true; shift;; *) echo "unsupported argument: $1" >&2; exit 2;; esac; done
[[ $smoke == true ]] && output_dir=${QA_SMOKE_OUTPUT_DIR:-${output_dir%_checkpoints}_smoke}
printf 'config=%s\nscene_root=%s\noutput_dir=%s\nmaster=%s:%s\n' "$config" "$scene_root" "$output_dir" "$master_addr" "$master_port"
[[ $dry_run == true ]] && exit 0
mkdir -p "$output_dir/logs" "$output_dir/pids"
for i in "${!hosts[@]}"; do
  host=${hosts[$i]}; iface=${ifaces[$i]}; log=$output_dir/logs/node$i.log
  smoke_env=""; [[ $smoke == true ]] && smoke_env="QA_EPOCHS=1 QA_MAX_TRAIN_SCENES=8 QA_MAX_VAL_SCENES=8"
  command="export NODE_RANK=$i MASTER_ADDR='$master_addr' MASTER_PORT='$master_port' NETWORK_IFACE='$iface' QA_CONFIG='$config' QA_OUTPUT_DIR='$output_dir' QA_SCENE_ROOT='$scene_root' $smoke_env; cd '$repo'; source /data/shared/shared/softwares/anaconda3/etc/profile.d/conda.sh; conda activate /data/shared/shared/envs/vjepa2; export PYTHONPATH='$repo'; exec '$repo/src/experiments/vjepa2_clevrer_dynamics_decoder_all_seed239/scripts/clevrer/train_qa_decoder_4node_worker.sh'"
  ssh user@$host "nohup bash -lc $(printf '%q' "$command") >'$log' 2>&1 < /dev/null & echo \$!" > "$output_dir/pids/node$i.pid"
done
echo "started QA training on 4 nodes; logs=$output_dir/logs"
