#!/usr/bin/env bash
# Run on the MODEL machine. Keeps seven independent policy servers alive.
set -euo pipefail

# 加载公共配置
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
server_defaults 0.0.0.0
init_batch

# 校验模型设备与路径
IFS=, read -r -a DEVICES_ARRAY <<< "${DEVICES}"
validate_devices
if [[ ${CHECKPOINT} == /path/to/* ]]; then
  echo "Set CHECKPOINT in evaluation/robotwin/common.sh" >&2
  exit 2
fi

# 创建日志目录并选择示范
LOG_ROOT="${LOG_ROOT:-${COSMOS_ROOT}/results/robotwin_servers/$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "${LOG_ROOT}"
LOG_ROOT=$(cd "${LOG_ROOT}" && pwd -P)
[[ ! -e ${LOG_ROOT}/pairs.json ]] || {
  echo "Use a fresh LOG_ROOT" >&2
  exit 2
}
extra=()
if [[ -n ${PAIR_MAP_JSON:-} ]]; then
  extra+=(--pair-map "${PAIR_MAP_JSON}")
fi
"${SERVER_PYTHON}" "${EVAL_DIR}/prepare_pairs.py" \
  --checkpoint "${CHECKPOINT}" \
  --manifest "${ICL_PAIR_MANIFEST}" \
  --output "${LOG_ROOT}" \
  "${extra[@]}"

# 按任务顺序启动七个服务
mapfile -t PAIRS < "${LOG_ROOT}/pairs.tsv"
((${#PAIRS[@]} == 7)) || exit 2
for i in "${!TASKS[@]}"; do
  IFS=$'\t' read -r task pair <<< "${PAIRS[i]}"
  [[ ${task} == "${TASKS[i]}" ]] || {
    echo "Task mapping mismatch" >&2
    exit 2
  }
  port=$((START_PORT + i))
  setsid bash "${EVAL_DIR}/launch_server.sh" "${task}" "${pair}" "${port}" "${DEVICES_ARRAY[i]}" \
    > "${LOG_ROOT}/${task}.log" 2>&1 &
  PIDS+=("$!")
  echo "${task}: port=${port}, device=${DEVICES_ARRAY[i]}, pid=$!"
done

# Watch startup and server death together; one failed server terminates the batch.
probe_host=${SERVER_BIND}
[[ ${probe_host} != 0.0.0.0 ]] || probe_host=127.0.0.1
setsid "${SERVER_PYTHON}" "${EVAL_DIR}/batch_tools.py" wait \
  --host "${probe_host}" \
  --port "${START_PORT}" \
  --timeout "${SERVER_START_TIMEOUT}" \
  --execute-steps 1 \
  --output "${LOG_ROOT}" &
probe_pid=$!
PIDS+=("${probe_pid}")
status=0
wait -n -p finished "${PIDS[@]}" || status=$?
if [[ ${finished} != "${probe_pid}" ]] || ((status)); then
  echo "Server startup failed; see ${LOG_ROOT}" >&2
  exit 1
fi

# 保持服务运行，任一服务退出时清理其余进程
unset 'PIDS[-1]'
echo "All seven services ready. Logs: ${LOG_ROOT}. Keep this launcher running; Ctrl-C stops all servers."
wait -n "${PIDS[@]}" || true
echo "A policy server exited; stopping the remaining services." >&2
exit 1
