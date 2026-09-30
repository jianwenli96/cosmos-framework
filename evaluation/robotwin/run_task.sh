#!/usr/bin/env bash
# Internal worker: TASK PAIR PORT MODEL_DEVICE SIM_DEVICE SAVE_ROOT.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
server_defaults
client_defaults

# 单个任务的参数
task=$1
pair=$2
port=$3
model_device=$4
sim_device=$5
save_root=$6

# 进程清理
server_pid=''
client_pid=''
cleanup() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${client_pid}" "${server_pid}"; do
    if [[ -n ${pid} ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${client_pid}" "${server_pid}"; do
    if [[ -n ${pid} ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
  exit "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 启动服务
bash "${EVAL_DIR}/launch_server.sh" "${task}" "${pair}" "${port}" "${model_device}" \
  > "${save_root}/logs/${task}_server.log" 2>&1 &
server_pid=$!

# 等待服务就绪
deadline=$((SECONDS + SERVER_START_TIMEOUT))
while true; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "${task}: server exited; see ${save_root}/logs/${task}_server.log" >&2
    exit 1
  fi
  if curl --fail --silent --max-time 2 "http://${HOST}:${port}/info" > "${save_root}/logs/${task}_info.json"; then
    break
  fi
  if ((SECONDS >= deadline)); then
    echo "${task}: server startup timed out" >&2
    exit 1
  fi
  sleep 2
done

# 执行测评
bash "${EVAL_DIR}/launch_client.sh" "${task}" "${port}" "${sim_device}" "${save_root}/${task}" \
  > "${save_root}/logs/${task}_client.log" 2>&1 &
client_pid=$!
wait "${client_pid}"
client_pid=''
echo "Finished ${task}"
