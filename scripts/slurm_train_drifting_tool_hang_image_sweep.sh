#!/bin/bash
#SBATCH --job-name=th-img-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/train_drifting_tool_hang_image_sweep_%j.out
#SBATCH --partition=agpu,vgpu
#SBATCH --constraint="1a100|1v100"
#SBATCH --exclude=c2110

# Usage: sbatch --export=GEN_PER_LABEL=4 scripts/slurm_train_drifting_tool_hang_image_sweep.sh

set -euo pipefail

GEN_PER_LABEL=${GEN_PER_LABEL:-8}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "gen_per_label: $GEN_PER_LABEL"

PROJECT_ROOT=$(pwd)
CONDA_ENV_NAME="robodiff"
CONTAINER="$HOME/qwenvl-2.5-cu121.sif"

mkdir -p slurm_logs

apptainer exec --nv --writable-tmpfs \
    --bind "$HOME:$HOME" \
    --bind /share/apps:/share/apps \
    "${CONTAINER}" bash -c "
source /share/apps/python/anaconda-3.14/etc/profile.d/conda.sh
cd ${PROJECT_ROOT}

conda activate \"${CONDA_ENV_NAME}\"

export LD_LIBRARY_PATH=\$HOME/.mujoco/mujoco210/bin:\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
export MUJOCO_PY_MUJOCO_PATH=\$HOME/.mujoco/mujoco210
export MUJOCO_GL=egl

echo '=== Launching Drifting Policy Training (Tool Hang Image, gen_per_label=${GEN_PER_LABEL}) ==='
python train.py \
    --config-dir=. \
    --config-name=drifting_tool_hang_image.yaml \
    policy.gen_per_label=${GEN_PER_LABEL} \
    training.rollout_every=25 \
    training.seed=42 \
    training.device=cuda:0 \
    training.resume=false \
    hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}_gpl${GEN_PER_LABEL}'
"

echo "Job finished at: $(date)"
