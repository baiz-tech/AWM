#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/data/shared/shyang/workspace/work/vjepa2-baiz}
python_bin=${PYTHON_BIN:-/data/shared/shared/envs/vjepa2/bin/python}
data_root=${LOCAL_DATA_ROOT:-/home/shyang/workspace/data/clevrer_train_decoder}
output_dir=${OUTPUT_DIR:?OUTPUT_DIR is required}
node_rank=${NODE_RANK:?NODE_RANK is required}
master_addr=${MASTER_ADDR:-10.22.129.130}
master_port=${MASTER_PORT:-29500}
num_gpus=${NUM_GPUS:-8}
iface=${NETWORK_IFACE:-enp14s0np0}

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-4}
export NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-4}
export NCCL_SOCKET_IFNAME=$iface
export GLOO_SOCKET_IFNAME=$iface
export NNODES=4

cd "$repo"
resume_args=()
[[ -n ${RESUME_CHECKPOINT:-} ]] && resume_args+=(--resume-checkpoint "$RESUME_CHECKPOINT")
limit_args=()
[[ -n ${MAX_TRAIN_SCENES:-} ]] && limit_args+=(--max-train-scenes "$MAX_TRAIN_SCENES")
[[ -n ${MAX_VALIDATION_SCENES:-} ]] && limit_args+=(--max-validation-scenes "$MAX_VALIDATION_SCENES")

exec "$python_bin" -m torch.distributed.run \
  --nnodes=4 --node_rank="$node_rank" \
  --master_addr="$master_addr" --master_port="$master_port" \
  --nproc_per_node="$num_gpus" \
  src/experiments/vjepa2_clevrer_dynamics_decoder_all_seed239/scripts/clevrer/train_structured_probe.py \
  --nonpredictive-cache-root "$data_root/latents_nonpredictive" \
  --predictive-cache-root "$data_root/latents_predictive" \
  --train-targets "$data_root/targets/targets_train.pt" \
  --validation-targets "$data_root/targets/targets_validation.pt" \
  --output-dir "$output_dir" \
  --epochs "${EPOCHS:-30}" --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-2}" --hidden-dim "${HIDDEN_DIM:-256}" \
  --num-heads "${NUM_HEADS:-8}" --num-slots "${NUM_SLOTS:-6}" \
  --slot-depth "${SLOT_DEPTH:-1}" --transition-depth "${TRANSITION_DEPTH:-1}" \
  --interaction-depth "${INTERACTION_DEPTH:-1}" --ffn-dim "${FFN_DIM:-1024}" \
  --dropout "${DROPOUT:-0.1}" --learning-rate "${LR:-2e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.04}" --grad-clip "${GRAD_CLIP:-1.0}" \
  --seed "${SEED:-239}" --log-every "${LOG_EVERY:-25}" \
  "${resume_args[@]}" "${limit_args[@]}"
