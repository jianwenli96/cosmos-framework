#!/usr/bin/env bash
# Run on the SIMULATOR machine; no checkpoint, manifest or HumanGen dependency.
set -euo pipefail

# 加载公共配置
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
client_defaults
init_batch

# 校验仿真设备与路径
IFS=, read -r -a DEVICES_ARRAY <<< "${SIM_DEVICES}"
validate_devices
if [[ ${ROBOTWIN_ROOT} == /path/to/* ]]; then
  echo "Set ROBOTWIN_ROOT in evaluation/robotwin/common.sh" >&2
  exit 2
fi

# 创建本次输出目录
SAVE_ROOT="${SAVE_ROOT:-${COSMOS_ROOT}/results/robotwin_clients/$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "${SAVE_ROOT}"
SAVE_ROOT=$(cd "${SAVE_ROOT}" && pwd -P)
[[ ! -e ${SAVE_ROOT}/services.json ]] || {
  echo "Use a fresh SAVE_ROOT" >&2
  exit 2
}
mkdir -p "${SAVE_ROOT}/logs"

# Validate all task-to-port mappings before starting any simulation.
setsid "${CLIENT_PYTHON}" "${EVAL_DIR}/batch_tools.py" wait \
  --host "${HOST}" \
  --port "${START_PORT}" \
  --timeout "${SERVER_START_TIMEOUT}" \
  --execute-steps "${EXECUTE_STEPS}" \
  --output "${SAVE_ROOT}" &
PIDS+=("$!")
wait "${PIDS[0]}"
PIDS=()

# 并行启动七个客户端
for i in "${!TASKS[@]}"; do
  task=${TASKS[i]}
  setsid bash "${EVAL_DIR}/launch_client.sh" "${task}" "$((START_PORT + i))" \
    "${DEVICES_ARRAY[i]}" "${SAVE_ROOT}/${task}" > "${SAVE_ROOT}/logs/${task}.log" 2>&1 &
  PIDS+=("$!")
  echo "${task}: server=${HOST}:$((START_PORT + i)), GPU=${DEVICES_ARRAY[i]}, pid=$!"
done

# 等待测评结束并汇总
status=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || status=1
done
PIDS=()
"${CLIENT_PYTHON}" "${EVAL_DIR}/batch_tools.py" summary \
  --output "${SAVE_ROOT}"
echo "Results: ${SAVE_ROOT}"
exit "${status}"
