#!/bin/bash
# Download BlockPush and Kitchen training data from Diffusion Policy data server
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data

echo "=== Downloading BlockPush data ==="
cd data
if [ ! -d "block_pushing" ]; then
    wget https://diffusion-policy.cs.columbia.edu/data/training/block_pushing.zip
    unzip block_pushing.zip && rm -f block_pushing.zip
    echo "BlockPush data downloaded to data/block_pushing/"
else
    echo "data/block_pushing/ already exists, skipping."
fi

echo ""
echo "=== Downloading Kitchen data ==="
if [ ! -d "kitchen" ]; then
    wget https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip
    unzip kitchen.zip && rm -f kitchen.zip
    echo "Kitchen data downloaded to data/kitchen/"
else
    echo "data/kitchen/ already exists, skipping."
fi

cd ..
echo ""
echo "=== Done ==="
echo "BlockPush zarr: data/block_pushing/multimodal_push_seed.zarr"
echo "Kitchen demos:  data/kitchen/kitchen_demos_multitask/"
echo "Kitchen init:   data/kitchen/all_init_qpos.npy, all_init_qvel.npy"
