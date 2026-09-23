#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_humangen_env.sh"
humangen_train_job "$@"

train_flags=()
[[ "${DRYRUN:-0}" != 1 ]] || train_flags+=(--dryrun)

# Merge worker stderr into stdout before torchrun captures rank 0.
exec "$PYTHON_BIN" -m torch.distributed.run --nnodes=1 --nproc_per_node="${NGPU:-8}" \
    --master_port="${MASTER_PORT:-50130}" --log-dir="$TRAIN_LOG_DIR" --redirects 0 --tee 0:1 \
    --no-python bash -c 'exec "$@" 2>&1' bash "$PYTHON_BIN" -u \
    -m cosmos_framework.scripts.train --sft-toml="examples/toml/sft_config/human_video_icl_${ICL_MODE}_edge.toml" "${train_flags[@]}" -- \
    "model.config.vlm_config.tokenizer.tokenizer_type=$COSMOS3_EDGE_PROCESSOR_PATH" \
    "${train_job_flags[@]}"
