#!/bin/bash
#SBATCH --job-name=ddpm-lift-image
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm_logs/ddpm_lift_image_%j.out
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

echo '=== DDPM Baseline: Lift Image (100 epochs) ==='
echo '=== If train_action_mse_error decreases and test/mean_score > 0 at epoch 50, data pipeline is OK ==='

# Generate absolute action dataset if it doesn't exist
if [ ! -f data/robomimic/datasets/lift/ph/image_abs.hdf5 ]; then
    echo '=== Converting to absolute actions ==='
    python -m diffusion_policy.scripts.robomimic_dataset_conversion \
        -i data/robomimic/datasets/lift/ph/image.hdf5 \
        -o data/robomimic/datasets/lift/ph/image_abs.hdf5
fi
python train.py \
    --config-dir=. \
    --config-name=ddpm_lift_image.yaml \
    training.seed=42 \
    training.device=cuda:0 \
    training.num_epochs=100 \
    training.resume=false \
    hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_ddpm_lift_image_baseline'
"

echo "Job finished at: $(date)"
