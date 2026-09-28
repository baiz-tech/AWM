#!/usr/bin/env bash
set -euo pipefail

# Launch the three independent preparation jobs from the jump host.
# node01: nonpredictive latents; node02: predictive latents; node05: targets.
repo=${REPO:-/data/shared/shyang/workspace/work/vjepa2-baiz}
out=${OUTPUT_DIR:-/data/shared/shyang/outputs/vjepa2-baiz/vjepa2_naive_probe_v6_decoder_probeTest1_clevrerAll}
env_dir=${ENV_DIR:-/data/shared/envs/vjepa2-312}
conda_root=${CONDA_ROOT:-/data/shared/shared/softwares/anaconda3}
dataset_root=${DATASET_ROOT:-/data/shared/datasets/CLEVRER}
encoder_ckpt=${ENCODER_CHECKPOINT:-/data/ABDUCTIVE-WORLD/pretrain/checkpoints/vith.pt}
predictor_ckpt=${PREDICTOR_CHECKPOINT:-/data/ABDUCTIVE-WORLD/pretrain/checkpoints/vith.pt}
config=${CONFIG:-$repo/recipe/vjepa2_naive_probe_v6_decoder_probeTest1_clevrerAll/configs/clevrer-fullpatch-structured-probe-v1-8gpu.yaml}
python_bin=${PYTHON_BIN:-$env_dir/bin/python}
window_starts=${WINDOW_STARTS:-0}
gpus=${NUM_GPUS:-8}
remote_user=${REMOTE_USER:-shyang}
mkdir -p "$out/logs" "$out/launcher_logs"

run_remote() {
  local name=$1 host=$2 mode=$3 script=$4 extra=${5:-}
  local iface=enp14s0np0
  [[ $host == 10.22.16.141 ]] && iface=enp85s0np0
  local log="$out/logs/${name}.log"
  local launcher_log="$out/launcher_logs/${name}.launcher.log"
  local command
  if [[ $script == latent ]]; then
    command="cd '$repo' && source '$conda_root/etc/profile.d/conda.sh' && conda activate '$env_dir' && export PYTHONPATH='$repo'\${PYTHONPATH:+:\$PYTHONPATH} NCCL_IB_DISABLE=1 NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4 NCCL_SOCKET_IFNAME=$iface GLOO_SOCKET_IFNAME=$iface DATA_EXPERIMENT_ROOT='$out' WORLD_CHECKPOINT='$predictor_ckpt' && DECODER_MODE='$mode' LATENT_CACHE_NAME='latents_$mode' WINDOW_STARTS='$window_starts' '$repo/recipe/vjepa2_naive_probe_v6_decoder_probeTest1_clevrerAll/scripts/clevrer/cache_full_latents.sh' --run --output-mode data"
  else
    command="cd '$repo' && source '$conda_root/etc/profile.d/conda.sh' && conda activate '$env_dir' && export PYTHONPATH='$repo' && '$python_bin' '$repo/recipe/vjepa2_naive_probe_v6_decoder_probeTest1_clevrerAll/scripts/clevrer/prepare_targets.py' --dataset-root '$dataset_root' --split train --output '$out/targets/targets_train.pt' --window-starts '$window_starts' --mode predictive && '$python_bin' '$repo/recipe/vjepa2_naive_probe_v6_decoder_probeTest1_clevrerAll/scripts/clevrer/prepare_targets.py' --dataset-root '$dataset_root' --split validation --output '$out/targets/targets_validation.pt' --window-starts '$window_starts' --mode predictive"
  fi
  printf '[%s] %s -> %s\n' "$name" "$host" "$log"
  ssh "$remote_user@$host" "nohup bash -lc $(printf '%q' "$command") >'$log' 2>&1 & echo \$!" >"$launcher_log"
}

if [[ ${1:-} == --dry-run ]]; then
  printf 'output=%s\nnonpredictive=node01\npredictive=node02\ntargets=node05\nencoder=%s\npredictor=%s\n' "$out" "$encoder_ckpt" "$predictor_ckpt"
  exit 0
fi

run_remote nonpredictive_latents 10.22.16.133 nonpredictive latent
run_remote predictive_latents 10.22.16.135 predictive latent
run_remote targets 10.22.16.140 predictive targets
printf 'launched three jobs under %s\n' "$out"
