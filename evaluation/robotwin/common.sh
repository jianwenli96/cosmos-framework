#!/usr/bin/env bash
# Shared configuration. Edit defaults here; environment variables override them.
set -euo pipefail

# 路径与公共运行参数
EVAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
COSMOS_ROOT=$(cd "${EVAL_DIR}/../.." && pwd -P)
export
export PYTHONPATH="${COSMOS_ROOT}:${PYTHONPATH:-}"
export

export SEED="${SEED:-0}"
export START_PORT="${START_PORT:-8000}"
export
export SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"

# 七任务设备映射
DEVICES="${DEVICES:-0,1,2,3,4,5,6}"
SIM_DEVICES="${SIM_DEVICES:-${DEVICES}}"
mapfile -t TASKS < "${EVAL_DIR}/tasks.txt"

# Called only on the model host (or by the combined local launcher).
server_defaults() {
  export
  # 模型、数据与解释器路径
  export CHECKPOINT="${CHECKPOINT:-/path/to/run/checkpoints/iter_000001000/model}"
  export
  export SERVER_PYTHON="${SERVER_PYTHON:-/mnt/sfs_turbo/public/apps/miniforge3/envs/cosmos-icl/bin/python}"
  export
  export HUMAN_GEN_ROOT="${HUMAN_GEN_ROOT:-/mnt/sfs_turbo/public/datasets/HumanGen}"
  export
  export ICL_PAIR_MANIFEST="${ICL_PAIR_MANIFEST:-/mnt/sfs_turbo/lijianwen/Datasets/cosmos_manifest/cosmos_humangen_manifest.json}"
  export
  export WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/sfs_turbo/public/ckpts/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
  export COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge}"
  export

  # 推理参数与服务监听地址
  export COSMOS_DEVICE="${COSMOS_DEVICE:-npu}"
  export
  export MODE="${MODE:-generator}"
  export STEPS="${STEPS:-20}"
  export
  export SERVER_BIND="${SERVER_BIND:-${1:-127.0.0.1}}"
}

# Called only on the simulator host (or by the combined local launcher).
client_defaults() {
  export
  # 仿真环境与服务地址
  export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/RoboTwin}"
  export
  export CLIENT_PYTHON="${CLIENT_PYTHON:-python}"
  export
  export HOST="${SERVER_IP:-${HOST:-127.0.0.1}}"
  export

  # 评测参数
  export TEST_NUM="${TEST_NUM:-100}"
  export EXECUTE_STEPS="${EXECUTE_STEPS:-16}"
  export
  export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
  export
  export INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-seen}"
  export EPISODE_OFFSET="${EPISODE_OFFSET:-0}"
  export

  # 录像与渲染
  export SAVE_VIDEO="${SAVE_VIDEO:-1}"
  export VIDEO_FPS="${VIDEO_FPS:-10}"
  export VIDEO_STRIDE="${VIDEO_STRIDE:-5}"
  export
  export SKIP_RENDER_CHECK="${SKIP_RENDER_CHECK:-0}"
  export

  # 尝试次数与超时
  export MAX_ATTEMPTS="${MAX_ATTEMPTS:-10000}"
  export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
  export
  export START_SEED="${START_SEED:-}"
}

# Opt in explicitly: sourcing defaults alone must not install process traps.
init_batch() {
  PIDS=()
  trap batch_cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  command -v setsid >/dev/null
}

batch_cleanup() {
  local status=$?
  trap - EXIT INT TERM

  for pid in "${PIDS[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done

  # Bound shutdown time even if a simulator ignores SIGTERM.
  for ((attempt=0; attempt<10; attempt++)); do
    local alive=0
    for pid in "${PIDS[@]}"; do
      if kill -0 -- "-${pid}" 2>/dev/null; then
        alive=1
      fi
    done
    ((alive)) || break
    sleep 1
  done

  for pid in "${PIDS[@]}"; do
    kill -KILL -- "-${pid}" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  exit "${status}"
}

validate_devices() {
  if ((${#DEVICES_ARRAY[@]} != 7)); then
    echo "Specify exactly seven device IDs (one per task)" >&2
    exit 2
  fi
  for device in "${DEVICES_ARRAY[@]}"; do
    [[ ${device} =~ ^[0-9]+$ ]] || {
      echo "Invalid device: ${device}" >&2
      exit 2
    }
  done
  [[ ${START_PORT} =~ ^[0-9]+$ ]] || exit 2
  ((START_PORT >= 1024 && START_PORT <= 55529)) || exit 2
}
