#!/usr/bin/env bash
set -euo pipefail
RUN=${RUN_DIR:?set RUN_DIR};NAME=${PROBE_NAME:-probe_target_only_v1};PORT=${MASTER_PORT:-29644};PY=${PYTHON_BIN:-python};OUT="$RUN/$NAME";[[ ! -e "$OUT" ]]||{ echo "output exists: $OUT" >&2;exit 2;};LOG="$RUN/${NAME}_train.log";mkdir -p "$RUN"
nohup "$PY" -m torch.distributed.run --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$PORT" --nproc_per_node=8 -m legacy.vjepa2_naive_probe_v5_decoder_physionpp2.scripts.physionpp.train_target_only --cache-root "$RUN/cache" --train-targets "$RUN/targets_target_only_data_v1.pt" --validation-targets "$RUN/targets_target_only_readout_data_v1.pt" --output-dir "$OUT" >"$LOG" 2>&1 & echo $! >"$RUN/${NAME}_train.pid";echo "started pid=$! log=$LOG"
