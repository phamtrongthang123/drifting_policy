# Drifting Policy for PushT (Visual)

A minimal fork of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) implementing
[Drifting Model](https://lambertae.github.io/projects/drifting/) on the simulated PushT vision task.
All other policies (IBC, BET, robomimic), tasks (kitchen, blockpush, robomimic), and real-robot code have been stripped out.
If the authors of Drifting Model release the robotic control experiment then I will stop developing this repo, or else I will explore with other settings until I can reproduce the reported scores. 

## Current results

| Batch size | Best mean score |
|------------|-----------------|
| 256        | ~0.78           |
| 64         | ~0.6x           |

The original Drifting Model paper reports higher scores in a plug-and-play setting.
But I couldn't make it to that score (0.86) after many trial and errors. So I think getting comparable numbers likely requires further hyperparameter tuning
(learning rate, EMA schedule, temperatures, number of epochs, etc.) beyond what the paper provides.

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

## Data

Download the PushT dataset into a `data/` directory:
```bash
mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm -f pusht.zip
cd ..
```

This should produce `data/pusht/pusht_cchi_v7_replay.zarr`.

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

Two SLURM scripts are provided in `scripts/`:
- `slurm_train_drifting.sh` -- 2-day job on A100
- `slurm_train_drifting_l1.sh` -- 6-hour short job on A100

```bash
sbatch scripts/slurm_train_drifting.sh
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
├── train.py                        # Training entry point
├── eval.py                         # Evaluation entry point
├── drifting_pusht_image.yaml       # Main config
├── setup.py                        # Package installer
├── conda_environment.yaml          # Conda env spec
├── scripts/
│   ├── slurm_train_drifting.sh
│   └── slurm_train_drifting_l1.sh
└── diffusion_policy/
    ├── workspace/                  # Training loop
    ├── policy/                     # Drifting policy + base class
    ├── model/
    │   ├── drifting/               # Drifting loss computation
    │   ├── diffusion/              # U-Net backbone, EMA
    │   ├── vision/                 # Observation encoder
    │   └── common/                 # Normalizer, LR scheduler, mixins
    ├── env_runner/                 # PushT evaluation runner
    ├── gym_util/                   # Async vector env, wrappers
    ├── common/                     # Replay buffer, sampler, utilities
    ├── real_world/                 # VideoRecorder (used by wrappers)
    ├── codecs/                     # Zarr image codec
    └── config/task/                # PushT image task config
```

## Quick explanation of the meat of my implementation

See [drifting_model_debug.md](drifting_model_debug.md) for notes on understanding my implementation on drifting model.
But mostly diffusion_policy/model/drifting/drifting_util.py is the key here. I faithfully "copied" the pseudo code in the paper out. This one is tested with the toy experiment in the provided [colab notebook](https://lambertae.github.io/projects/drifting/) so I am 99% confidence that it is correct.    

## Acknowledgements

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) -- base codebase
- [Drifting](https://lambertae.github.io/projects/drifting/) -- drifting policy method
- [`ConditionalUnet1D`](./diffusion_policy/model/diffusion/conditional_unet1d.py) adapted from [Planning with Diffusion](https://github.com/jannerm/diffuser)

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for details.
