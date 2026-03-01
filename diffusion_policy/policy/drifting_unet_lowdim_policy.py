from typing import Dict
import torch
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.drifting.drifting_util import compute_drifting_loss

class DriftingUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
            model: ConditionalUnet1D,
            horizon,
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            obs_as_global_cond=True,
            temperatures=[0.02, 0.05, 0.2],
            **kwargs):
        super().__init__()
        assert obs_as_global_cond
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.temperatures = temperatures
        self.kwargs = kwargs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'obs' in obs_dict
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        global_cond = nobs[:,:To].reshape(B, -1)

        noise = torch.randn(size=(B, T, Da), device=device, dtype=dtype)
        timesteps = torch.zeros((B,), device=device, dtype=torch.long)
        naction_pred = self.model(noise, timesteps, global_cond=global_cond)

        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs']
        nactions = nbatch['action']
        batch_size = nactions.shape[0]

        global_cond = nobs[:,:self.n_obs_steps].reshape(batch_size, -1)

        noise = torch.randn(nactions.shape, device=nactions.device)
        timesteps = torch.zeros((batch_size,), device=nactions.device, dtype=torch.long)
        pred_actions = self.model(noise, timesteps, global_cond=global_cond)

        x = pred_actions.reshape(batch_size, -1)
        y_pos = nactions.reshape(batch_size, -1)
        y_neg = x

        loss, metrics = compute_drifting_loss(x, y_pos, y_neg, temperatures=self.temperatures)
        return loss, metrics
