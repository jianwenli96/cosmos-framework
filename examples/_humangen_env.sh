#!/usr/bin/env bash
set -euo pipefail

FRAMEWORK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FRAMEWORK_ROOT"

# icl mode
ICL_MODE="${ICL_MODE:-generator}"
[[ "$ICL_MODE" == generator || "$ICL_MODE" == reasoner ]] || { echo 'ICL_MODE must be generator or reasoner' >&2; exit 1; }

# python path
export PYTHON_BIN="${PYTHON_BIN:-/mnt/sfs_turbo/public/apps/miniforge3/envs/cosmos-icl/bin/python}"

# dataset path
export HUMAN_GEN_ROOT="${HUMAN_GEN_ROOT:-/mnt/sfs_turbo/public/datasets/HumanGen}"
export ICL_PAIR_MANIFEST="${ICL_PAIR_MANIFEST:-/mnt/sfs_turbo/lijianwen/Datasets/cosmos_manifest/cosmos_humangen_manifest.json}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-"$FRAMEWORK_ROOT"/train_outputs}"

# Keep torchrun logs beside the training artifacts, including job overrides.
humangen_train_job() {
    local project=cosmos3_action group=humangen name="humangen_${ICL_MODE}" arg
    train_job_flags=("job.project=$project" "job.group=$group" "job.name=$name" "$@")
    for arg in "${train_job_flags[@]}"; do
        case "$arg" in
            job.project=*) project="${arg#*=}" ;;
            job.group=*) group="${arg#*=}" ;;
            job.name=*) name="${arg#*=}" ;;
        esac
    done
    TRAIN_LOG_DIR="${IMAGINAIRE_OUTPUT_ROOT}/${project}/${group}/${name}/logs"
}

# checkpoint path
export COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge-DCP}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/sfs_turbo/public/ckpts/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"

# environment variables
export TOKENIZERS_PARALLELISM=false
export COSMOS_DEVICE="${COSMOS_DEVICE:-npu}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$FRAMEWORK_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# check if the shared manifest exists
if [[ ! -f "$ICL_PAIR_MANIFEST" ]]; then
    echo "Prepare the shared manifest first: $PYTHON_BIN -m cosmos_framework.scripts.prepare_humangen_pairs --root $HUMAN_GEN_ROOT --output $ICL_PAIR_MANIFEST --sources robotwin,agibot,robocoin,robomind,interna1,oxe" >&2
    exit 1
fi

# check the output folder
"$PYTHON_BIN" -c 'import sys; from cosmos_framework.data.generator.action.datasets.humangen_dataset import safe_output; safe_output(sys.argv[1], sys.argv[2])' "$HUMAN_GEN_ROOT" "$IMAGINAIRE_OUTPUT_ROOT"
