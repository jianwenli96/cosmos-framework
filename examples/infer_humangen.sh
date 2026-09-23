#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_humangen_env.sh"

exec "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="${NGPU:-8}" --master_port="${MASTER_PORT:-50131}" \
    -m cosmos_framework.scripts.infer_icl_policy --mode "$ICL_MODE" "$@"
