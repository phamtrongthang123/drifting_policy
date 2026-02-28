#!/bin/bash
#SBATCH --job-name=train-drifting-can-image
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/train_drifting_can_image_%j.out
#SBATCH --partition=vgpu
#SBATCH --constraint=1v100

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

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

echo '=== Launching Drifting Policy Training (Can Image) ==='
python train.py \
    --config-dir=. \
    --config-name=drifting_can_image.yaml \
    training.seed=42 \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'
"

echo "Job finished at: $(date)"
