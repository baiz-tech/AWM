#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../../../.." && pwd)
python_bin=${PYTHON_BIN:-/data/shared/shared/envs/vjepa2/bin/python}
data_root=${DATA_ROOT:-/home/shyang/workspace/data/clevrer_train_decoder}
output_root=${OUTPUT_ROOT:-/data/shared/shyang/outputs/vjepa2-baiz/vjepa2_naive_probe_v5_decoder_clevrerAll/clevrer_qa_decoder_all/scene_outputs}
qa_root=${QA_ROOT:-/data/shared/shyang/outputs/vjepa2-baiz/vjepa2_naive_probe_v5_decoder_clevrerAll/clevrer_qa_decoder_all}
checkpoint=${DECODER_CHECKPOINT:?set DECODER_CHECKPOINT to decoder/probe best.pt}
gpus=${NUM_GPUS:-8}
resume_flag=()
[[ ${RESUME:-0} == 1 ]] && resume_flag+=(--resume)

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
for mode in nonpredictive predictive; do
  if [[ $mode == predictive ]]; then
    cache_root=${PREDICTIVE_CACHE_ROOT:-$qa_root/latents_predictive_32_159}
    output_mode=predictive_32_159
  else
    cache_root=${NONPREDICTIVE_CACHE_ROOT:-$data_root/latents_nonpredictive}
    output_mode=nonpredictive
  fi
  [[ -d "$cache_root" ]] || { echo "missing cache: $cache_root" >&2; exit 1; }
  for split in train validation; do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
      "$python_bin" -m torch.distributed.run \
      --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
      --master_port="${EXPORT_PORT:-29740}" --nproc_per_node="$gpus" \
      "$repo/src/experiments/vjepa2_clevrer_dynamics_decoder_all_seed239/scripts/clevrer/export_scene_decoder_outputs.py" \
      --cache-root "$cache_root" --checkpoint "$checkpoint" \
      --output-root "$output_root" --split "$split" --mode "$output_mode" \
      --batch-size "${EXPORT_BATCH_SIZE:-1}" "${resume_flag[@]}"
  done
done
