#!/usr/bin/env bash
# Evaluate all seven unique held-out tasks; one model server per active task.
set -euo pipefail

# All defaults live in common.sh; preserve this local serial/batched entry point.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
server_defaults
client_defaults

# 校验配置和设备
SAVE_ROOT="${SAVE_ROOT:-${COSMOS_ROOT}/results/robotwin/$(date +%Y%m%d_%H%M%S)_$$}"
for name in CHECKPOINT ROBOTWIN_ROOT CLIENT_PYTHON; do
  if [[ ${!name} == /path/to/* ]]; then
    echo "请先修改 common.sh 中的 ${name} 默认路径: ${!name}" >&2
    exit 2
  fi
done
command -v curl >/dev/null
command -v setsid >/dev/null
# Device IDs are physical IDs; pass comma-separated values, e.g. 0,1,2,3.
IFS=, read -r -a MODEL_DEVICES <<< "${DEVICES}"
IFS=, read -r -a SIM_DEVICES <<< "${SIM_DEVICES}"
if ((${#MODEL_DEVICES[@]} != ${#SIM_DEVICES[@]})); then
  echo "DEVICES and SIM_DEVICES must have equal lengths" >&2
  exit 2
fi
for device in "${MODEL_DEVICES[@]}" "${SIM_DEVICES[@]}"; do
  [[ ${device} =~ ^[0-9]+$ ]] || {
    echo "Invalid device ID: ${device}" >&2
    exit 2
  }
done
[[ ${START_PORT} =~ ^[0-9]+$ && ${SERVER_START_TIMEOUT} =~ ^[1-9][0-9]*$ ]] || exit 2
if ((START_PORT < 1024 || START_PORT + ${#MODEL_DEVICES[@]} + 10000 > 65535)); then
  echo "START_PORT must leave room for HTTP and distributed initialization ports" >&2
  exit 2
fi

# 准备输出目录与示范映射
mkdir -p "${SAVE_ROOT}"
SAVE_ROOT=$(cd "${SAVE_ROOT}" && pwd -P)
# Avoid overwriting previous task results.
if [[ -e ${SAVE_ROOT}/pairs.json ]]; then
  echo "SAVE_ROOT already contains a run; choose a fresh directory: ${SAVE_ROOT}" >&2
  exit 2
fi
extra=()
if [[ -n ${PAIR_MAP_JSON:-} ]]; then
  extra+=(--pair-map "${PAIR_MAP_JSON}")
fi
"${SERVER_PYTHON}" "${EVAL_DIR}/prepare_pairs.py" \
  --manifest "${ICL_PAIR_MANIFEST}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${SAVE_ROOT}" \
  "${extra[@]}"
mkdir -p "${SAVE_ROOT}/logs"

# Each worker owns a session, so signals also reach its Python/SAPIEN children.
init_batch

export CHECKPOINT HUMAN_GEN_ROOT ICL_PAIR_MANIFEST ROBOTWIN_ROOT
export SERVER_PYTHON CLIENT_PYTHON MODE SEED TEST_NUM STEPS EXECUTE_STEPS TASK_CONFIG
export HOST SERVER_BIND SERVER_START_TIMEOUT

# 按可用设备数量分批执行
mapfile -t PAIRS < "${SAVE_ROOT}/pairs.tsv"
width=${#MODEL_DEVICES[@]}
status=0
for ((base=0; base<${#PAIRS[@]}; base+=width)); do
  PIDS=()
  for ((slot=0; slot<width && base+slot<${#PAIRS[@]}; slot++)); do
    IFS=$'\t' read -r task pair <<< "${PAIRS[base+slot]}"
    port=$((START_PORT + slot))
    if curl --silent --max-time 1 "http://${HOST}:${port}/info" >/dev/null; then
      echo "Port ${port} already has an HTTP service; choose another START_PORT" >&2
      exit 1
    fi
    echo "Starting ${task}: model device ${MODEL_DEVICES[slot]}, simulation GPU ${SIM_DEVICES[slot]}, port ${port}"
    setsid bash "${EVAL_DIR}/run_task.sh" "${task}" "${pair}" "${port}" \
      "${MODEL_DEVICES[slot]}" "${SIM_DEVICES[slot]}" "${SAVE_ROOT}" &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" || status=1
  done
  PIDS=()
done

# 汇总所有任务结果
"${SERVER_PYTHON}" "${EVAL_DIR}/batch_tools.py" summary \
  --output "${SAVE_ROOT}"
echo "Results and logs: ${SAVE_ROOT}"
exit "${status}"
