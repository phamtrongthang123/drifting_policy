# Drifting Model Debugging Guide

This document serves as a reference for debugging and understanding the Drifting Model implementation in this codebase.

## 📂 Key Files
- **`diffusion_policy/model/drifting/drifting_util.py`**: **Core Logic.** Contains the drifting field calculation (`compute_V`) and the training objective (`compute_drifting_loss`). This is where normalization ($S_j, \lambda_j$) and temperatures are handled.
   - The compute V is exactly the same as Algo 2. 
   - **Status**: Verified correct against paper equations and toy demos.
   - **Note**: This implementation enables **Drift Normalization** (Appendix A.8). Consequently, the training loss will remain stable around **1.0** (due to normalization). To track convergence, monitor the **`lambda_j`** metric (Drift Magnitude), which should decrease over time.

- **`diffusion_policy/policy/drifting_unet_hybrid_image_policy.py`**: **Policy Wrapper.** Manages the observation encoding and interfaces the U-Net with the drifting loss.

## cloned from the diffusion version. not much changes here. 
- **`diffusion_policy/workspace/train_drifting_unet_hybrid_workspace.py`**: **Training Loop.** Handles the epoch-by-epoch logic, optimization, and logging (e.g., `train_action_mse_error`). There is not much here to compare. This is just the diffusion version cloned. 
- **`drifting_pusht_image.yaml`**: **Main Config.** Defines hyper-parameters like `temperatures`, `batch_size`, and `rollout_every`. There is not much here to compare with the paper. 

## 📓 Reference Demos (`drifting_reference/demo/`)
These notebooks provide isolated environments to verify the core math and logic.

- **`drifting_model_demo_original.ipynb`**: **Toy 2D Demo (Unnormalized).**
    - Uses the raw drift field ($||\mathbf{V}||^2$ loss).
    - Loss decreases to 0 as distributions match.
    - Useful for verifying basic correctness of the vector field logic.

- **`drifting_model_demo_with_norm.ipynb`**: **Toy 2D Demo (Normalized).**
    - Implements the **Drift Normalization** (Appendix A.8) used in the main codebase.
    - **Key Behavior**: The MSE Loss stays flat (~1.0), while the `lambda` (drift magnitude) decreases. This matches the behavior of our main training loop.

- **`drifting_model_demo.py`**: A python script version of the toy demo.

- **`diffusion_policy_state_pusht_demo.py`** / **`diffusion_policy_vision_pusht_demo.py`**: 
    - Reference implementations of standard **Diffusion Policy** on PushT tasks.
    - Serve as baselines for architecture and data processing comparison.

## 📈 Monitoring Progress
- **`train/mean_score`**: The primary evaluation metric (average of max rewards).
- **`train_action_mse_error`**: A proxy for how well the 1-step generator maps noise to the action space.
- **`train/drifting_lambda_T{t}`**: **Crucial for Convergence.** Represents the magnitude of the drift field. It should trend downward as the model learns.
- **Batch Size Sensitivity**: Drifting Models require a good estimate of the current distribution ($q$). If the batch size is too small, the "repulsion" field will be noisy, causing instability.