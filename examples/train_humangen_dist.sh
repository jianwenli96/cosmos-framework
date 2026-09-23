#!/usr/bin/env bash
set -euo pipefail

NNODES="${NNODES:-${MA_NUM_HOSTS:-}}"
NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-}}"
worker_hosts="${VC_WORKER_HOSTS:-}"
MASTER_ADDR="${MASTER_ADDR:-${worker_hosts%%,*}}"
NGPU="${NGPU:-${MA_NUM_GPUS:-16}}"

: "${NNODES:?Set NNODES or MA_NUM_HOSTS}" "${NODE_RANK:?Set NODE_RANK or VC_TASK_INDEX}" "${MASTER_ADDR:?Set MASTER_ADDR or VC_WORKER_HOSTS}"

export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"

source "$(dirname -- "${BASH_SOURCE[0]}")/_humangen_env.sh"
humangen_train_job "$@"

train_flags=()
[[ "${DRYRUN:-0}" != 1 ]] || train_flags+=(--dryrun)

# Persist every rank on every node; separate nodes on the shared filesystem.
# Torchrun tee selects local ranks; only node 0 contains global rank 0.
node_log_dir="$TRAIN_LOG_DIR/node_${NODE_RANK}"
mkdir -p "$node_log_dir"
echo "Training logs for node $NODE_RANK: $node_log_dir"
log_flags=(--log-dir="$node_log_dir" --redirects 3 --tee 0)
if [[ "$NODE_RANK" == 0 ]]; then
    log_flags=(--log-dir="$node_log_dir" --redirects 3 --tee 0:1)
fi

# Merge worker stderr into stdout so each rank's traceback stays with its logs.
exec "$PYTHON_BIN" -m torch.distributed.run --nnodes="$NNODES" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-50130}" --nproc_per_node="$NGPU" "${log_flags[@]}" \
    --no-python bash -c 'exec "$@" 2>&1' bash "$PYTHON_BIN" -u \
    -m cosmos_framework.scripts.train --sft-toml="examples/toml/sft_config/human_video_icl_${ICL_MODE}_edge.toml" "${train_flags[@]}" -- \
    "model.config.vlm_config.tokenizer.tokenizer_type=$COSMOS3_EDGE_PROCESSOR_PATH" \
    "${train_job_flags[@]}"
