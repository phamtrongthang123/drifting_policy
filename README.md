# Drifting Policy

A minimal fork of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) implementing
[Drifting Model in Generative Modeling via Drifting](https://lambertae.github.io/projects/drifting/) on simulated robotic control tasks.
All other policies (IBC, BET), real-robot code, and unrelated tasks have been stripped out.

## Paper targets (Table 3) vs current results

Numbers are **mean success rate** (higher is better). Drifting Policy uses **NFE=1** (single forward pass at inference).

The drifting loss is a **faithful line-by-line port** of the [official JAX implementation](https://lambertae.github.io/projects/drifting/), verified by numerical tests (7/7 pass, rtol=2e-4 vs JAX). Key config: `batch_size=64`, `gen_per_label=8`.

| Task | Setting | Diffusion Policy (NFE=100) | **Drifting Policy (paper)** | Our result |
|------|---------|---------------------------|----------------------------|------------|
| **Can** | Visual | 0.97 | **0.99** | **0.98** (epoch 50) ✅ |
| **PushT** | Visual | 0.84 | **0.86** | **0.86** (epoch 100) ✅ |
| **Lift** | Visual | 1.00 | **1.00** | **1.00** (epoch 50) ✅ |
| **ToolHang** | Visual | 0.73 | 0.67 | 0.00 (epoch 0, running) |


### Low-dim (state-based) tasks

| Task | Setting | Diffusion Policy | Drifting Policy (paper) | Our status |
|------|---------|-----------------|------------------------|------------|
| **PushT** | State | **0.91** | 0.86 | **0.871** (epoch 50) ✅ |
| Can | State | 0.96 | **0.98** | **0.98** (epoch 50) ✅ |
| ToolHang | State | 0.30 | **0.38** | **0.84** (epoch 50) ✅ |
| Lift | State | 0.98 | **1.00** | **1.00** (epoch 50) ✅ |

## Installation

**Prerequisites** (Ubuntu 20.04, Nvidia GPU):
```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

Create the conda environment:
```bash
conda env create -f conda_environment.yaml
conda activate robodiff
```

Install the package (needed for `diffusion_policy.*` imports):
```bash
pip install -e .
```

### MuJoCo and robosuite 

PushT only needs `pygame` (already in conda env). The robomimic tasks need MuJoCo 2.1.0 + `mujoco-py` + a specific robosuite fork.

**1. MuJoCo 2.1.0** (not the newer `mujoco` pip package — we need the old `mujoco_py` bindings):
```bash
mkdir -p ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
rm mujoco210-linux-x86_64.tar.gz
# -> ~/.mujoco/mujoco210/
```

Add to your `~/.bashrc` (or SLURM scripts):
```bash
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export MUJOCO_GL=egl   # for headless GPU rendering
```

**2. mujoco-py** (builds Cython extensions on first import):
```bash
pip install free-mujoco-py==2.1.6
```

**3. robosuite** — must use the `cheng-chi/robosuite` fork at the `offline_study` branch. This fork has ToolHang support and is compatible with `mujoco_py`. Do **not** use `robosuite>=1.4` (uses new mujoco bindings, breaks EGL rendering on clusters).
```bash
pip install robosuite@https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz
```

**4. robomimic** (dataset loading):
```bash
pip install robomimic==0.2.0
```

Verify the install:
```bash
python -c "import mujoco_py; import robosuite; print(f'mujoco_py={mujoco_py.__version__}, robosuite={robosuite.__version__}')"
# Expected: mujoco_py=2.0.2.13, robosuite=1.2.0
```

## Data

**PushT** (Zarr format):
```bash
mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm -f pusht.zip && cd ..
# -> data/pusht/pusht_cchi_v7_replay.zarr
```

**Robomimic tasks** (Lift, Can, ToolHang) via the robomimic download script:
```bash
python -m robomimic.scripts.download_datasets \
    --download_dir data/robomimic/datasets \
    --tasks lift can tool_hang \
    --dataset_types ph \
    --hdf5_types image
# -> data/robomimic/datasets/{lift,can,tool_hang}/ph/image.hdf5
# Sizes: lift ~800 MB, can ~2 GB, tool_hang ~62 GB
```

## Training

```bash
conda activate robodiff
wandb login          # optional, for experiment tracking

python train.py \
    --config-dir=. \
    --config-name=drifting_pusht_image.yaml \
    training.seed=42 \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

Override batch size (default 512 in config):
```bash
python train.py \
    --config-dir=. \
    --config-name=drifting_pusht_image.yaml \
    dataloader.batch_size=256 \
    training.seed=42 \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

Checkpoints and logs are written to `data/outputs/`.
The policy is evaluated every 50 epochs; `test/mean_score` is logged to wandb.

### SLURM

```bash
sbatch scripts/slurm_train_drifting.sh              # PushT (A100, 2-day)
sbatch scripts/slurm_train_drifting_l1.sh           # PushT quick test (A100, 6-hour)
sbatch scripts/slurm_train_drifting_lift_image.sh   # Lift visual (V100, 2-day)
sbatch scripts/slurm_train_drifting_can_image.sh    # Can visual (V100, 2-day)
sbatch scripts/slurm_train_drifting_tool_hang_image.sh  # ToolHang visual (V100, 2-day)
```

## Evaluation

Evaluate a saved checkpoint:
```bash
python eval.py \
    --checkpoint data/outputs/.../checkpoints/latest.ckpt \
    --output_dir data/eval_output \
    --device cuda:0
```

Results are written to `data/eval_output/eval_log.json`.

## Project structure

```
.
├── train.py                              # Training entry point
├── eval.py                               # Evaluation entry point
├── drifting_pusht_image.yaml             # PushT visual config
├── drifting_lift_image.yaml              # Lift visual config
├── drifting_can_image.yaml               # Can visual config
├── drifting_tool_hang_image.yaml         # ToolHang visual config
├── drifting_*_lowdim.yaml               # Low-dim (state) configs
├── setup.py                              # Package installer
├── conda_environment.yaml                # Conda env spec
├── scripts/
│   ├── slurm_train_drifting.sh           # PushT (A100, 2-day)
│   ├── slurm_train_drifting_l1.sh        # PushT quick (A100, 6-hour)
│   ├── slurm_train_drifting_lift_image.sh
│   ├── slurm_train_drifting_can_image.sh
│   └── slurm_train_drifting_tool_hang_image.sh
└── diffusion_policy/
    ├── workspace/                        # Training loop
    ├── policy/                           # Drifting policy + base class
    ├── model/
    │   ├── drifting/                     # Drifting loss computation
    │   ├── diffusion/                    # U-Net backbone, EMA
    │   ├── vision/                       # Observation encoder
    │   └── common/                       # Normalizer, LR scheduler, mixins
    ├── env_runner/                       # Evaluation runners (PushT, robomimic)
    ├── gym_util/                         # Async vector env, wrappers
    ├── common/                           # Replay buffer, sampler, utilities
    ├── real_world/                       # VideoRecorder (used by wrappers)
    ├── codecs/                           # Zarr image codec
    └── config/task/                      # Task configs
```

## Implementation notes

The core drifting loss lives in `diffusion_policy/model/drifting/drifting_util.py`. It is a **line-by-line port** of the official JAX implementation (`drifting/drift_loss.py`), verified by `tests/test_drift_loss_port.py` (7 numerical tests, rtol=2e-4 vs JAX reference).

Key details:
- **`gen_per_label=8`**: multiple noise samples per observation (matching official `train.py`). With `gen_per_label=1`, the self-mask zeros the drift force.
- **Dot-product cdist** (not `torch.cdist`): matches the official kernel exactly.

## Acknowledgements

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) -- base codebase
- [Drifting](https://lambertae.github.io/projects/drifting/) -- drifting policy method
- [`ConditionalUnet1D`](./diffusion_policy/model/diffusion/conditional_unet1d.py) adapted from [Planning with Diffusion](https://github.com/jannerm/diffuser)

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for details.
