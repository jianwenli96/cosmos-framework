#!/usr/bin/env bash
# Usage: launch_client.sh TASK PORT GPU OUTPUT
set -euo pipefail

# 加载配置并检查参数
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
client_defaults

: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the RoboTwin checkout}"

if [[ $# != 4 ]]; then
  echo "Usage: $0 TASK PORT GPU OUTPUT" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$3"

# 指令、录像与随机种子选项
extra=(
  --instruction-type "${INSTRUCTION_TYPE}"
  --episode-offset "${EPISODE_OFFSET}"
)

if [[ ${SAVE_VIDEO} == 1 ]]; then
  extra+=(
    --save-video
    --video-fps "${VIDEO_FPS}"
    --video-stride "${VIDEO_STRIDE}"
  )
else
  extra+=(--no-save-video)
fi

if [[ ${SKIP_RENDER_CHECK} == 1 ]]; then
  extra+=(--skip-render-check)
fi

if [[ -n ${START_SEED:-} ]]; then
  extra+=(--start-seed "${START_SEED}")
fi

# 启动单任务仿真
exec "${CLIENT_PYTHON}" -m cosmos_framework.scripts.eval_robotwin \
  --robotwin-root "${ROBOTWIN_ROOT}" \
  --task "$1" \
  --task-config "${TASK_CONFIG}" \
  --server "http://${HOST}:$2" \
  --output "$4" \
  --episodes "${TEST_NUM}" \
  --execute-steps "${EXECUTE_STEPS}" \
  --seed "${SEED}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --timeout "${REQUEST_TIMEOUT}" \
  "${extra[@]}"
