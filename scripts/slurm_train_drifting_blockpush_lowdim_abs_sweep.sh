#!/bin/bash
#SBATCH --job-name=bp-abs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/train_drifting_blockpush_abs_%A_%a.out
#SBATCH --partition=agpu,vgpu
#SBATCH --constraint="1a100|1v100"
#SBATCH --exclude=c2110
#SBATCH --array=0-3

# BlockPush lowdim with abs_action=True, sweep gen_per_label
# Usage: sbatch scripts/slurm_train_drifting_blockpush_lowdim_abs_sweep.sh

set -euo pipefail

GPL_VALUES=(2 4 8 16)
GEN_PER_LABEL=${GPL_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "abs_action=true, gen_per_label: $GEN_PER_LABEL"

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

echo '=== BlockPush Lowdim ABS (gpl=${GEN_PER_LABEL}) ==='
python train.py \
    --config-dir=. \
    --config-name=drifting_blockpush_lowdim.yaml \
    task.dataset.zarr_path=data/block_pushing/multimodal_push_seed_abs.zarr \
    task.env_runner.abs_action=true \
    policy.gen_per_label=${GEN_PER_LABEL} \
    training.rollout_every=1 \
    training.seed=42 \
    training.device=cuda:0 \
    training.resume=false \
    hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}_abs_gpl${GEN_PER_LABEL}'
"

echo "Job finished at: $(date)"
