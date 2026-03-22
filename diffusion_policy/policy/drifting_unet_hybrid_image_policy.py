from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.drifting.drifting_util import drift_loss

class DriftingUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            temperatures=[0.02, 0.05, 0.2],
            per_timestep_loss=False,
            bc_coeff=0.0,
            gen_per_label=8,
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
        
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.temperatures = temperatures
        self.per_timestep_loss = per_timestep_loss
        self.bc_coeff = bc_coeff
        self.gen_per_label = gen_per_label
        self.kwargs = kwargs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1)
        
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
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        G = self.gen_per_label

        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

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
