"""
action_model.py

"""
from InternVLA.model.modules.action_model.DiT_modules.models import DiT
from InternVLA.model.modules.action_model import create_diffusion
from .DiT_modules import gaussian_diffusion as gd

import torch
from torch import nn

# Create model sizes of ActionModels
def DiT_S(**kwargs): # TODO 不能在这里这样定义， 要统一 成为 config, 保证内部参数一致
    return DiT(depth=6, token_size=384, num_heads=4, **kwargs)
def DiT_B(**kwargs):
    return DiT(depth=12, token_size=768, num_heads=12, **kwargs)
def DiT_L(**kwargs):
    return DiT(depth=24, token_size=1024, num_heads=16, **kwargs)

# Model size
DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L}

# Create ActionModel
class ActionModel(nn.Module):
    def __init__(self, 
                 action_hidden_dim, 
                 model_type, 
                 in_channels, 
                 future_action_window_size, 
                 past_action_window_size,
                 diffusion_steps = 100,
                 noise_schedule = 'squaredcos_cap_v2'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = noise_schedule, diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.token_size = action_hidden_dim # 这是 QFormer 的大小， 这里会混乱的情况
        self.net = DiT_models[model_type](
                                        in_channels=in_channels, 
                                        class_dropout_prob = 0.1, 
                                        learn_sigma = learn_sigma, 
                                        future_action_window_size = future_action_window_size, 
                                        past_action_window_size = past_action_window_size
                                        )

    # Given condition z and ground truth token x, compute loss
    def loss(self, x, z):
        # x: [B, T, C] tensor of ground truth tokens TODO 确定 z 的shape 
        # z: [B, L, D_action] tensor of condition tokens (e.g., latent_action: [B, 64, 768])
        # sample random noise and timestep
        noise = torch.randn_like(x) # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device= x.device)

        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise)

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, z)

        assert noise_pred.shape == noise.shape == x.shape
        # Compute L2 loss
        loss = ((noise_pred - noise) ** 2).mean() # TODO 是否这里导致了 batch 并不能加速？
        # Optional: loss += loss_vlb

        return loss

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion = create_diffusion(timestep_respacing = "ddim"+str(ddim_step), 
                                               noise_schedule = self.noise_schedule,
                                               diffusion_steps = self.diffusion_steps, 
                                               sigma_small = True, 
                                               learn_sigma = False
                                               )
        return self.ddim_diffusion
    
def get_action_model(model_typ="DiT-B", config=None):
    """
    根据配置创建 ActionModel 实例
    :param config: 包含模型参数的配置字典或对象
    :return: ActionModel 实例
    """
    action_model_cfg = config.framework.action_model

    model_type = action_model_cfg.action_model_type
    action_hidden_dim = action_model_cfg.action_hidden_dim
    action_dim = action_model_cfg.action_dim
    future_action_window_size = action_model_cfg.future_action_window_size
    past_action_window_size =  action_model_cfg.past_action_window_size
    
    return ActionModel(
        model_type=model_type,  # 模型类型，例如 'DiT-B'
        action_hidden_dim=action_hidden_dim,  # 动作隐藏维度
        in_channels=action_dim,  # 输入通道数
        future_action_window_size=future_action_window_size,  # 未来动作窗口大小
        past_action_window_size=past_action_window_size,  # 过去动作窗口大小
    )