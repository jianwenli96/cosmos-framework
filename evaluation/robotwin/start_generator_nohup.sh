#!/usr/bin/env bash
# Start the seven generator policy servers in the background.
# Usage: bash evaluation/robotwin/start_generator_nohup.sh
set -eo pipefail

# CANN's environment script may reference unset variables.
source "${CANN_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}" || exit 1
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)
cd "${REPO_ROOT}"

export CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/train_outputs/cosmos3_action/humangen/humangen_generator_window_17/checkpoints/iter_000050000/model}"
export MODE=generator
export COSMOS_DEVICE=npu
export DEVICES="${DEVICES:-0,1,2,3,4,5,6}"
export SERVER_BIND="${SERVER_BIND:-0.0.0.0}"
export START_PORT="${START_PORT:-8000}"
export STEPS="${STEPS:-20}"
export SEED="${SEED:-0}"

source "${SCRIPT_DIR}/common.sh"
server_defaults
IFS=, read -r -a DEVICES_ARRAY <<< "${DEVICES}"
validate_devices
[[ -d ${CHECKPOINT} && -f ${CHECKPOINT}/.metadata ]] || {
  echo "Missing model checkpoint: ${CHECKPOINT}" >&2
  exit 1
}
[[ -x ${SERVER_PYTHON} ]] || {
  echo "Invalid SERVER_PYTHON: ${SERVER_PYTHON}" >&2
  exit 1
}

export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/results/robotwin_servers/generator_$(date +%Y%m%d_%H%M%S)_$$}"
# Require a fresh directory so an earlier run's logs and PID are never overwritten.
mkdir -p "$(dirname "${LOG_ROOT}")"
mkdir "${LOG_ROOT}"
LOG_ROOT=$(cd "${LOG_ROOT}" && pwd -P)

nohup bash "${SCRIPT_DIR}/launch_server_multigpus.sh" > "${LOG_ROOT}/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "${launcher_pid}" > "${LOG_ROOT}/launcher.pid"

sleep 1
if ! kill -0 "${launcher_pid}" 2>/dev/null; then
  echo "Launcher exited early; see ${LOG_ROOT}/launcher.log" >&2
  exit 1
fi

echo "Generator server launcher started in background (PID ${launcher_pid})."
echo "Wait for 'All seven services ready' in the launcher log."
printf 'View logs: tail -f %q\n' "${LOG_ROOT}/launcher.log"
echo "Task logs: ${LOG_ROOT}/<task>.log"
echo "Stop all seven services: kill -TERM ${launcher_pid}"
