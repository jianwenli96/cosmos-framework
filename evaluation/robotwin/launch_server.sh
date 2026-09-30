#!/usr/bin/env bash
# Usage: launch_server.sh TASK PAIR_ID PORT DEVICE
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
server_defaults

: "${CHECKPOINT:?Set CHECKPOINT to checkpoints/iter_XXXXXXXXX/model}"
: "${HUMAN_GEN_ROOT:?Set HUMAN_GEN_ROOT}"
: "${ICL_PAIR_MANIFEST:?Set ICL_PAIR_MANIFEST}"

# 校验参数
if [[ $# != 4 ]]; then
  echo "Usage: $0 TASK PAIR_ID PORT DEVICE" >&2
  exit 2
fi

# 选择模型设备
if [[ ${COSMOS_DEVICE:-cuda} == npu ]]; then
  export
  export ASCEND_RT_VISIBLE_DEVICES="$4"
else
  export
  export CUDA_VISIBLE_DEVICES="$4"
fi

# Each task is an independent single-device server, even inside a torchrun shell.
export
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export
export MASTER_PORT=$(($3 + 10000))

# 启动单任务推理服务
exec "${SERVER_PYTHON}" -m cosmos_framework.scripts.action_policy_server_robotwin \
  --mode "${MODE}" \
  --checkpoint "${CHECKPOINT}" \
  --root "${HUMAN_GEN_ROOT}" \
  --manifest "${ICL_PAIR_MANIFEST}" \
  --split test \
  --pair-id "$2" \
  --task "$1" \
  --host "${SERVER_BIND}" \
  --port "$3" \
  --steps "${STEPS}" \
  --seed "${SEED}"
