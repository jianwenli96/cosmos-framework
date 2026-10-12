#!/usr/bin/env bash
# Run on the H20 simulator host after syncing the current client code.
# Usage with the SSH reverse tunnel: bash evaluation/robotwin/start_client_nohup.sh
# Tunnel mapping: H20 127.0.0.1:28000-28006 -> model host 127.0.0.1:8000-8006.
# Direct connection: START_PORT=8000 bash evaluation/robotwin/start_client_nohup.sh MODEL_SERVER_IP
# Or: SERVER_IP=MODEL_SERVER_IP bash evaluation/robotwin/start_client_nohup.sh
# For an existing SSH tunnel, explicitly pass 127.0.0.1 and its START_PORT.
set -euo pipefail

if (($# > 1)); then
  echo "Usage: $0 [MODEL_SERVER_IP] (defaults to SSH tunnel at 127.0.0.1:28000)" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)
cd "${REPO_ROOT}"

export SERVER_IP="${1:-${SERVER_IP:-127.0.0.1}}"
export START_PORT="${START_PORT:-28000}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/a-glj-ws/RoboTwin}"
export CLIENT_PYTHON="${CLIENT_PYTHON:-/home/ubuntu/a-glj-ws/miniconda3/envs/robotwin/bin/python}"
export SIM_DEVICES="${SIM_DEVICES:-0,1,0,1,0,1,0}"
export TEST_NUM="${TEST_NUM:-1}"
export SEED="${SEED:-0}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export COSMOS_RT_RAY_GUARD="${COSMOS_RT_RAY_GUARD:-1}"
export PYTHONUNBUFFERED=1

source "${SCRIPT_DIR}/common.sh"
client_defaults
IFS=, read -r -a DEVICES_ARRAY <<< "${SIM_DEVICES}"
validate_devices
[[ -f ${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml ]] || {
  echo "Missing RoboTwin task config: ${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml" >&2
  exit 1
}
command -v "${CLIENT_PYTHON}" >/dev/null || {
  echo "Invalid CLIENT_PYTHON: ${CLIENT_PYTHON}" >&2
  exit 1
}
command -v setsid >/dev/null

export SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/results/robotwin_clients/generator_$(date +%Y%m%d_%H%M%S)_$$}"
# Keep logs, PID and evaluation results together; never overwrite an earlier run.
mkdir -p "$(dirname "${SAVE_ROOT}")"
mkdir "${SAVE_ROOT}"
SAVE_ROOT=$(cd "${SAVE_ROOT}" && pwd -P)

nohup bash "${SCRIPT_DIR}/launch_client_multigpus.sh" > "${SAVE_ROOT}/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "${launcher_pid}" > "${SAVE_ROOT}/launcher.pid"

sleep 1
if ! kill -0 "${launcher_pid}" 2>/dev/null; then
  echo "Launcher exited early; see ${SAVE_ROOT}/launcher.log" >&2
  exit 1
fi

echo "Client launcher started in background (PID ${launcher_pid})."
echo "Waiting for seven services at ${SERVER_IP}:${START_PORT}-$((START_PORT + 6))."
printf 'View logs: tail -f %q\n' "${SAVE_ROOT}/launcher.log"
echo "Task logs: ${SAVE_ROOT}/logs/<task>.log"
echo "Results and videos: ${SAVE_ROOT}"
echo "Stop clients: kill -TERM ${launcher_pid} (remote model servers stay running)"
