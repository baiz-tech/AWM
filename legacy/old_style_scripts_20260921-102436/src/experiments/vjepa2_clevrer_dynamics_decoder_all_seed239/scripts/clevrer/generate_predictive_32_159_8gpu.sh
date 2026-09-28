#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../../../.." && pwd)
python_bin=${PYTHON_BIN:-/data/shared/shared/envs/vjepa2/bin/python}
config=${CONFIG:-$repo/src/experiments/vjepa2_clevrer_dynamics_decoder_all_seed239/configs/clevrer-fullpatch-structured-probe-v1-8gpu.yaml}
checkpoint=${VJEPA2_CHECKPOINT:-/data/shared/shared/models/vjepa2/weights/vith.pt}
output_root=${OUTPUT_ROOT:-/data/shared/shyang/outputs/vjepa2-baiz/vjepa2_naive_probe_v5_decoder_clevrerAll/clevrer_qa_decoder_all}
gpus=${NUM_GPUS:-8}
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"

for split in train validation; do
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
    "$python_bin" -m torch.distributed.run \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --master_port="${PREDICTIVE_PORT:-29730}" --nproc_per_node="$gpus" \
    "$repo/src/experiments/vjepa2_clevrer_dynamics_decoder_all_seed239/scripts/clevrer/cache_full_latents.py" \
    --config "$config" --split "$split" --checkpoint "$checkpoint" \
    --output-dir "$output_root/latents_predictive_32_159" \
    --window-starts 32 --mode predictive --batch-size "${CACHE_BATCH_SIZE:-1}" \
    --num-workers "${CACHE_NUM_WORKERS:-2}" --resume
done
