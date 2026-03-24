#!/bin/bash
#SBATCH --job-name=bp-bsweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/train_drifting_blockpush_bsweep_%A_%a.out
#SBATCH --partition=agpu,vgpu
#SBATCH --constraint="1a100|1v100"
#SBATCH --exclude=c2110
#SBATCH --array=0-5

# BlockPush lowdim sweep: batch_size × diffusion_step_embed_dim × lr_scheduler
# Usage: sbatch scripts/slurm_train_drifting_blockpush_lowdim_bsweep.sh

set -euo pipefail

# Sweep configs:        bs   embed  scheduler
#   0: match original   256  256    cosine
#   1: original + bs512 512  256    cosine
#   2: isolate embed    256  256    constant_with_warmup
#   3: isolate sched    256  128    cosine
#   4: original small   128  256    cosine
#   5: big + warmup     512  256    constant_with_warmup
BATCH_SIZES=(256 512 256 256 128 512)
EMBED_DIMS=(256 256 256 128 256 256)
SCHEDULERS=(cosine cosine constant_with_warmup cosine cosine constant_with_warmup)

BATCH_SIZE=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}
EMBED_DIM=${EMBED_DIMS[$SLURM_ARRAY_TASK_ID]}
SCHEDULER=${SCHEDULERS[$SLURM_ARRAY_TASK_ID]}
GEN_PER_LABEL=${GEN_PER_LABEL:-8}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "batch_size: $BATCH_SIZE, embed_dim: $EMBED_DIM, scheduler: $SCHEDULER, gpl: $GEN_PER_LABEL"

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

echo '=== BlockPush Lowdim (bs=${BATCH_SIZE}, embed=${EMBED_DIM}, sched=${SCHEDULER}, gpl=${GEN_PER_LABEL}) ==='
python train.py \
    --config-dir=. \
    --config-name=drifting_blockpush_lowdim.yaml \
    dataloader.batch_size=${BATCH_SIZE} \
    policy.model.diffusion_step_embed_dim=${EMBED_DIM} \
    policy.gen_per_label=${GEN_PER_LABEL} \
    training.lr_scheduler=${SCHEDULER} \
    training.rollout_every=1 \
    training.seed=42 \
    training.device=cuda:0 \
    training.resume=false \
    hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}_bs${BATCH_SIZE}_e${EMBED_DIM}_${SCHEDULER}'
"

echo "Job finished at: $(date)"
