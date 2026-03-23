from typing import Dict
import torch
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.drifting.drifting_util import drift_loss

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
            per_timestep_loss=False,
            gen_per_label=8,
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
        self.per_timestep_loss = per_timestep_loss
        self.gen_per_label = gen_per_label
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
        G = self.gen_per_label

        global_cond = nobs[:,:self.n_obs_steps].reshape(batch_size, -1)

        # Generate G samples per observation (official: gen_per_label)
        global_cond_rep = global_cond.repeat_interleave(G, dim=0)  # [B*G, cond_dim]
        noise = torch.randn(batch_size * G, nactions.shape[1], nactions.shape[2], device=nactions.device)
        timesteps = torch.zeros((batch_size * G,), device=nactions.device, dtype=torch.long)
        pred_all = self.model(noise, timesteps, global_cond=global_cond_rep)  # [B*G, T, D]
        pred_actions = pred_all.reshape(batch_size, G, nactions.shape[1], nactions.shape[2])  # [B, G, T, D]

        R_list = tuple(self.temperatures)

        if self.per_timestep_loss:
            T_horizon = nactions.shape[1]
            total_loss = 0
            accumulated_metrics = {}
            for t in range(T_horizon):
                gen_t = pred_actions[:, :, t, :]           # [B, G, D]
                pos_t = nactions[:, t, :].unsqueeze(1)     # [B, 1, D]
                loss_t, info_t = drift_loss(gen_t, pos_t, R_list=R_list)
                total_loss = total_loss + loss_t.mean()
                for k, v in info_t.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v.item() / T_horizon
            loss = total_loss / T_horizon
            all_metrics = accumulated_metrics
        else:
            gen = pred_actions.reshape(batch_size, G, -1)          # [B, G, T*D]
            pos = nactions.reshape(batch_size, 1, -1)              # [B, 1, T*D]
            loss, info = drift_loss(gen, pos, R_list=R_list)
            loss = loss.mean()
            all_metrics = {k: v.item() for k, v in info.items()}

        return loss, all_metrics
