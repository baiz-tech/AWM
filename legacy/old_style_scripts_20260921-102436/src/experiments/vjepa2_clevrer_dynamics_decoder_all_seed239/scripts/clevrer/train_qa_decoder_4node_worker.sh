#!/usr/bin/env bash
set -euo pipefail
repo=${REPO:-/data/shared/shyang/workspace/work/vjepa2-baiz}; python_bin=${PYTHON_BIN:-/data/shared/shared/envs/vjepa2/bin/python}
export NCCL_IB_DISABLE=1 NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4 NCCL_SOCKET_IFNAME=${NETWORK_IFACE:-enp14s0np0} GLOO_SOCKET_IFNAME=${NETWORK_IFACE:-enp14s0np0}
export NNODES=4
epochs=${QA_EPOCHS:-400}; max_train=${QA_MAX_TRAIN_SCENES:-}; max_val=${QA_MAX_VAL_SCENES:-}
args=(--config "$QA_CONFIG" --decoder-scene-root "$QA_SCENE_ROOT" --output-dir "$QA_OUTPUT_DIR" --epochs "$epochs")
[[ -n $max_train ]] && args+=(--max-train-scenes "$max_train")
[[ -n $max_val ]] && args+=(--max-val-scenes "$max_val")
cd "$repo"
exec "$python_bin" -m torch.distributed.run --nnodes=4 --node_rank="${NODE_RANK:?}" --master_addr="${MASTER_ADDR:?}" --master_port="${MASTER_PORT:-29510}" --nproc_per_node=8 src/data/clevrer/train_qa.py "${args[@]}"
