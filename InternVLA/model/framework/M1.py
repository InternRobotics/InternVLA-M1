"""
InternVLA-M1.py

"""
from typing import List

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch, json
import torch.nn as nn
import numpy as np
from PIL import Image
import re
from InternVLA.model.framework.share_tools import read_mode_config

from InternVLA.training.trainer_utils import initialize_overwatch
logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from InternVLA.model.framework.share_tools import dict_to_namespace



from InternVLA.model.modules.vlm.QWen2_5 import get_qwen2_5_interface
from InternVLA.model.modules.projector.QFormer import get_layerwise_qformer
from InternVLA.model.modules.action_model.DiTActionHeader import get_action_model
from InternVLA.model.modules.dino_model.dino import get_dino_model
from InternVLA.model.framework.base_framework import baseframework
from InternVLA.training.trainer_utils.metrics import resize_images


class InternVLA_M1(baseframework):
    def __init__(
        self,
        config: Optional[dict] = None,  # @Jinhui TODO 这里应该是config, 但是现在是直接传入参数
        # norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config

        # TODO 全部转 全局config, 要面向对象编程
        self.qwen_vl_interface = get_qwen2_5_interface(config=self.config) 
        self.layer_qformer = get_layerwise_qformer(config=self.config) # @Jinhui 一般来说 人们喜欢总分结构， 但是有讨厌递归， 实验framework 下面就不能太总分了
        self.action_model = get_action_model(config=self.config)
        self.dino_encoder = get_dino_model(backone_name=getattr(self.config.framework.dino, "dino_backbone", "dinov2_vits14")) 
        self.dino_pro = nn.Linear( # 后期要重新考虑
            in_features=self.dino_encoder.num_channels,  # DINO 输出的特征维度  
            out_features=self.qwen_vl_interface.model.config.hidden_size)
        
        
        # TODO 为什么要在这个位置开始 看到 这些？--> 去思考， framework level 用户其他看到什么， 需要看到什么
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size

        # self.all_module_keys = auto_get_module_keys(self) #  TODO 这个是trainer的 funx， 或许是多余的
        # self.norm_stats = norm_stats # 这个是 inference 时候用到的， 不应该是放到这个位置？
        # self.use_ema = config.framework.action_model.use_ema


    def forward( # TODO 需要将 loss 计算分离出来
        self, # 只面对最原始的 data exmaples, 为了可读性，这里还是要写成显示的参数
        examples: List[dict] = None,  # 这里的 examples 是指原始的输入数据
        **kwargs,  # 👈 敏捷代码的灵活性， 允许任何形式的传参数
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        # @Jinhui TBD TODO  这里太长了，需要封装
        # 1. prepare input
        # 2. vlm forward
        # 3. action forward
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples] #label [B， len, 7]

        # Step 2: QWenVL 输入格式
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions = instructions) # @Jinhui TODO 再考虑一下这里的分支分流应该有.py控制还是由 if else
        
        with torch.autocast("cuda", dtype=torch.bfloat16): # @Jinhui TODO 这里的 dtype 是不是应该是config里面的
            qwenvl_outputs = self.qwen_vl_interface( # 都是local的参数变化， 不要写到config, 但是为了保持可复现，应该有个默认的 yaml
                **qwen_inputs, # 兼容性和可读性的 trade off
                output_attentions=False, # Flash attention 还不确定是否支持返回attention， 官方代码有bug
                output_hidden_states=True,
                return_dict=True,
                )
            pass

        # Step 1: 使用 DINO imaga processing 是什么？
        image_tensors = self.dino_encoder.prepare_dino_input(batch_images) # 
        B = len(batch_images)
        dino_features = self.dino_encoder(image_tensors)  # DINO 输出为 [B*num_view, token, dim]

        # 提取 DINO 的特征
        dino_encoded_features = dino_features.reshape(B, -1, dino_features.shape[-1])  # [B, num_view * token, dim]
        dino_encoded_features = self.dino_pro(dino_encoded_features) # [B, num_view * token, hidden_size]


        with torch.autocast("cuda", dtype=torch.float32):
            start_layer = self.config.framework.layer_qformer.qformer_start_layer if self.config else -6  # @Jinhui TODO 这里应该是config
            end_layer = self.config.framework.layer_qformer.qformer_end_layer if self.config else -1  # @Jinhui TODO 这里应该是config
            condition_features = qwenvl_outputs.hidden_states[start_layer:end_layer]
            
            # 给 DiT 添加来自dino 的输入
            cat_conditions = []
            for layer_index in range(len(condition_features)):
                layer_features = condition_features[layer_index]  # [B, n_qformer_token, D]
                layer_features = torch.cat([layer_features, dino_encoded_features], dim=1)  # [B, n_qformer_token + num_view * token, D] 
                cat_conditions.append(layer_features)

            action_latent_feature = self.layer_qformer(cat_conditions) # --> [B, 64, D_action]
    
            # actions = torch.stack([torch.tensor(a) for a in actions], dim=0).to(action_latent_feature.device)  # [B, chunk, 7] @Jinhui TODO to tensor 的逻辑可以放到 transform 里面
            # 先将 actions 转换为单个 NumPy 数组，再转换为 PyTorch 张量
            actions = torch.tensor(np.array(actions), device=action_latent_feature.device)  # [B, chunk, 7] TODO to tensor 的逻辑可以放到 transform 里面
            actions_future = actions[:, -(self.future_action_window_size+1):, :]
            
            # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
            repeated_diffusion_steps = self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
            action_latent_feature = action_latent_feature.repeat(repeated_diffusion_steps, 1, 1)  # [repeated_diffusion_steps*B, T, D_action]
            # Action model forward and compute loss # 这里功能有点 越俎代庖 TODO 将loss 集中到 main module中统一处理
            action_loss = self.action_model.loss(actions_repeated, action_latent_feature) # TODO loss 应该放到另一个函数
        return {"action_loss": action_loss}

    @torch.inference_mode() # @Jinhui DEBUG 临时取消
    def predict_action( # TODO 之后要和CoT 方案同步
        self, batch_images: List[List[Image.Image]], # B * List of PIL Image as [view1, view2]
        instructions: List[str], 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param batch_images: a batch of PIL Image as B* [view1, view2, ... ]
        @param instructions: Task instruction string 
        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        # align obs and lang
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)  # list of PIL RGB for one instruction
        instructions = [instruction.lower()  for instruction in instructions] # @Jinhui TODO 这里是为了兼容旧的格式， 需要考虑是否要将lang 变成 list
        

        inferface_inputs =  self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions) # @Jinhui TODO add instruction to qwenvl inputs
        qwen_inputs = inferface_inputs
        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`

        # Generate cognition feature through vlm
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # fmt: off
            qwenvl_outputs = self.qwen_vl_interface( # TODO 这里之后要用generation func
                input_ids=qwen_inputs.input_ids,
                attention_mask=qwen_inputs.attention_mask,
                pixel_values=qwen_inputs.pixel_values, # [512, 1176] 石斛没有 B,  
                image_grid_thw =qwen_inputs.image_grid_thw, # 2* [1,16,16] --> 512 = 16*16*2, 1176 = (224/16)^2 * 3 * 2 @JinhuiYE TODO 这个需要找Qwen 的官方文档验证
                # labels= qwen_inputs.input_ids.clone(),
                output_hidden_states=True, 
                return_dict=True,
            ) # generation 拿不到前面token 的信息，考虑使用 forward?

            B = len(batch_images)
            image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
            # print("image_tensors shape: ", image_tensors.shape) # [B, 3, 224, 224]
            dino_features = self.dino_encoder(image_tensors)  # DINO 输出为 OrderedDict


        with torch.autocast("cuda", dtype=torch.float32):
            # 提取 DINO 的特征
            B = dino_features.shape[0]
            dino_encoded_features = dino_features.view(B, -1,dino_features.shape[-1])  # [B, num_view * token, dim]
            dino_encoded_features = self.dino_pro(dino_encoded_features) # B, 256, D


            start_layer = self.config.framework.layer_qformer.qformer_start_layer if self.config else -6  # @Jinhui TODO 这里应该是config
            end_layer = self.config.framework.layer_qformer.qformer_end_layer if self.config else -1  # @Jinhui TODO 这里应该是config
            condition_features = qwenvl_outputs.hidden_states[start_layer:end_layer]
            
            # 给 DiT 添加来自dino 的输入
            cat_conditions = []
            for layer_index in range(len(condition_features)):
                layer_features = condition_features[layer_index]  # [B, n_qformer_token, D]
                layer_features = torch.cat([layer_features, dino_encoded_features], dim=1)  # [B, n_qformer_token + num_view * token, D] 
                cat_conditions.append(layer_features)

            action_latent_feature = self.layer_qformer(cat_conditions) # --> [B, 64, D_action]
    
            using_cfg = cfg_scale > 1.0

            model_dtype = next(self.action_model.net.parameters()).dtype
            B = action_latent_feature.shape[0]

            # Sample random noise
            noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=action_latent_feature.device).to(model_dtype)  #[B, T, D]

            # Setup classifier-free guidance:
            if using_cfg:
                noise = torch.cat([noise, noise], 0) #[2,16,7]
                uncondition = self.action_model.net.z_embedder.uncondition # [64, 768]
                uncondition_shape = uncondition.shape
                uncondition = uncondition.unsqueeze(0)  #[1, 64, D]
                uncondition = uncondition.expand(B, uncondition_shape[0], uncondition_shape[1]) #[B, n_qformer_token, D] # 
                z = torch.cat([action_latent_feature, uncondition], 0) # [2, 64, 768] TODO check 看看 training的时候是剁手
                cfg_scale = cfg_scale
                model_kwargs = dict(z=z, cfg_scale=cfg_scale)
                sample_fn = self.action_model.net.forward_with_cfg
            else:
                model_kwargs = dict(z=action_latent_feature)
                sample_fn = self.action_model.net.forward
            
            # DDIM Sampling
            if use_ddim and num_ddim_steps is not None:
                if self.action_model.ddim_diffusion is None: #@JinhuiYE =TODO check, shape 上没问题， 就不知道traine / infer 和内部操作是否有问题了
                    self.action_model.create_ddim(ddim_step=num_ddim_steps)
                samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=action_latent_feature.device,
                                                                    eta=0.0
                                                                    )
            else:
                # DDPM Sampling
                samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                        noise.shape, 
                                                                        noise, 
                                                                        clip_denoised=False,
                                                                        model_kwargs=model_kwargs,
                                                                        progress=False,
                                                                        device=action_latent_feature.device
                                                                        )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            normalized_actions = samples.cpu().numpy()

        return  {"normalized_actions": normalized_actions} # [B, T, action_dim]



def build_model_framework(config: dict = {}) -> InternVLA_M1:

    model = InternVLA_M1(config=config)
    return model




if __name__ == "__main__":
    from omegaconf import OmegaConf
    # 模型参数
    import debugpy
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client() 
    samples = {}

    config_yaml = "llavavla/conf/qwenvla_cotrain.yaml"
    cfg = OmegaConf.load(config_yaml)

    # try get model
    model_framework = build_model_framework(cfg)
    print(model_framework)

    # try forward model
    # can be fake sample， but here get from dataloader for simpler

    from InternVLA.dataloader.lerobot_datasets_oxe import get_vla_dataset, collate_fn

    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )
    
    from tqdm import tqdm

    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        batch
        break
    
    model_framework(batch)
