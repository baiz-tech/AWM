#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/data/shared/user/workspace/work/vjepa2-baiz}
worker=$repo/recipe/vjepa2_naive_probe_v6_decoder_probeTest4_clevrerAll/scripts/clevrer/train_structured_probe_4node_worker.sh
checker=$repo/recipe/vjepa2_naive_probe_v6_decoder_probeTest4_clevrerAll/scripts/clevrer/check_local_decoder_data.py
python_bin=${PYTHON_BIN:-/data/shared/envs/vjepa2-312/bin/python}
local_data_root=${LOCAL_DATA_ROOT:-/home/user/workspace/data/clevrer_train_decoder}
output_dir=${OUTPUT_DIR:-/data/shared/user/outputs/vjepa2-baiz/vjepa2_naive_probe_v6_decoder_probeTest4_clevrerAll/clevrer_train_decoder_run}
master_addr=${MASTER_ADDR:-192.0.2.10}
master_port=${MASTER_PORT:-29500}
remote_user=${REMOTE_USER:-user}
smoke=false
dry_run=false
resume_checkpoint=${RESUME_CHECKPOINT:-}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-test) smoke=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --resume)
      resume_checkpoint=${2:-}
      [[ -n $resume_checkpoint ]] || { echo "--resume requires a checkpoint" >&2; exit 2; }
      shift 2
      ;;
    *) echo "unsupported argument: $1" >&2; exit 2 ;;
  esac
done

if [[ $smoke == true ]]; then
  output_dir=${SMOKE_OUTPUT_DIR:-${output_dir%_run}_smoke}
fi
nodes=(worker-a worker-b worker-c worker-d)
hosts=(192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14)
ifaces=(eth0 eth0 eth0 eth1)

printf 'output_dir=%s\ndata_root=%s\nmaster=%s:%s\nsmoke=%s\nresume=%s\n' \
  "$output_dir" "$local_data_root" "$master_addr" "$master_port" "$smoke" "${resume_checkpoint:-none}"
for index in "${!nodes[@]}"; do
  printf '%s host=%s node_rank=%d iface=%s\n' \
    "${nodes[$index]}" "${hosts[$index]}" "$index" "${ifaces[$index]}"
done
[[ $dry_run == true ]] && exit 0

[[ -x $worker && -f $checker ]] || { echo "missing worker/checker scripts" >&2; exit 1; }
if [[ -e $output_dir/latest.pt && -z $resume_checkpoint ]]; then
  echo "output already has latest.pt; pass --resume $output_dir/latest.pt or use OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$output_dir/logs" "$output_dir/pids" "$output_dir/preflight"

for index in "${!nodes[@]}"; do
  node=${nodes[$index]}
  host=${hosts[$index]}
  echo "preflight $node"
  ssh "$remote_user@$host" \
    "test -x '$python_bin' && test -x '$worker' && test -d '$local_data_root' && test \"\$(nvidia-smi --list-gpus | wc -l)\" -ge 8 && ! pgrep -f '[t]rain_structured_probe.py' >/dev/null && '$python_bin' '$checker' --data-root '$local_data_root' --output '$output_dir/preflight/$node.json' >/dev/null"
done

for index in "${!nodes[@]}"; do
  node=${nodes[$index]}
  host=${hosts[$index]}
  iface=${ifaces[$index]}
  log=$output_dir/logs/$node.log
  pid_file=$output_dir/pids/$node.pid
  smoke_env=""
  [[ $smoke == true ]] && smoke_env="EPOCHS=1 MAX_TRAIN_SCENES=64 MAX_VALIDATION_SCENES=64"
  resume_env=""
  [[ -n $resume_checkpoint ]] && resume_env="RESUME_CHECKPOINT='$resume_checkpoint'"
  command="NODE_RANK=$index NETWORK_IFACE='$iface' MASTER_ADDR='$master_addr' MASTER_PORT='$master_port' LOCAL_DATA_ROOT='$local_data_root' OUTPUT_DIR='$output_dir' $smoke_env $resume_env '$worker'"
  ssh "$remote_user@$host" \
    "nohup bash -lc $(printf '%q' "$command") >'$log' 2>&1 </dev/null & echo \$!" >"$pid_file"
  echo "started $node pid=$(cat "$pid_file") log=$log"
done
