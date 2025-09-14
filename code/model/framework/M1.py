"""
InternVLA-M1.py

"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from code.training.trainer_utils import initialize_overwatch
logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from code.model.framework.base_framework import baseframework
from code.model.modules.vlm.QWen2_5 import get_qwen2_5_interface
from code.model.modules.projector.QFormer import get_layerwise_qformer
from code.model.modules.action_model.DiTActionHeader import get_action_model
from code.model.modules.dino_model.dino import get_dino_model
from code.training.trainer_utils.metrics import resize_images


class InternVLA_M1(baseframework):
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_qwen2_5_interface(config=self.config) 
        self.layer_qformer = get_layerwise_qformer(config=self.config) # @Jinhui 一般来说 人们喜欢总分结构， 但是有讨厌递归， 实验framework 下面就不能太总分了
        self.action_model = get_action_model(config=self.config)
        self.dino_encoder = get_dino_model(backone_name=getattr(self.config.framework.dino, "dino_backbone", "dinov2_vits14")) 
        self.dino_pro = nn.Linear(
            in_features=self.dino_encoder.num_channels,  # DINO 输出的特征维度  
            out_features=self.qwen_vl_interface.model.config.hidden_size)
        
        
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size

    def forward(
        self,
        examples: List[dict] = None,  # 这里的 examples 是指原始的输入数据
        **kwargs,  
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss). ask code assistant to get intro"""

        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples] #label [B， len, 7]

        # Step 1: QWenVL 输入格式
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions = instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface( 
                **qwen_inputs, 
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                )
            pass

        # Step 2: DINO Forward
        image_tensors = self.dino_encoder.prepare_dino_input(batch_images) # 
        B = len(batch_images)
        dino_features = self.dino_encoder(image_tensors)  # DINO 输出为 [B*num_view, token, dim]
        dino_encoded_features = dino_features.reshape(B, -1, dino_features.shape[-1])  # [B, num_view * token, dim]
        dino_encoded_features = self.dino_pro(dino_encoded_features) # [B, num_view * token, hidden_size]


        # Step 3: aggregation condition for Action expert
        start_layer = self.config.framework.layer_qformer.qformer_start_layer
        end_layer = self.config.framework.layer_qformer.qformer_end_layer
        condition_features = qwenvl_outputs.hidden_states[start_layer:end_layer]
        
        cat_conditions = []
        for layer_index in range(len(condition_features)):
            layer_features = condition_features[layer_index]  # [B, n_qformer_token, D]
            layer_features = torch.cat([layer_features, dino_encoded_features], dim=1)  # [B, n_qformer_token + num_view * token, D] 
            cat_conditions.append(layer_features)

        action_condition = self.layer_qformer(cat_conditions) # --> [B, 64, D_action]


        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):

            # here is a tips to accelerate training speed, by repeating each sample for several times @ref to CogACT
            actions = torch.tensor(np.array(actions), device=action_condition.device)  # [B, chunk, 7]
            actions_future = actions[:, -(self.future_action_window_size+1):, :]
            
            # tips: Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
            repeated_diffusion_steps = self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
            action_condition = action_condition.repeat(repeated_diffusion_steps, 1, 1)  # [repeated_diffusion_steps*B, T, D_action]
            
            # DiT noise add and predict
            noise_pred, noise, timestep = self.action_model(actions_repeated, action_condition)
            
            # perdition loss
            action_loss = self.action_model.loss(noise_pred, noise)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
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
        @return Unnormalized (continuous) action vector.

        # align forward prediction is very important for robot
        """
        # align obs and lang
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = [instruction.lower()  for instruction in instructions]
        

        inferface_inputs =  self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions) # @Jinhui TODO add instruction to qwenvl inputs
        qwen_inputs = inferface_inputs

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_hidden_states=True, 
                return_dict=True,
            )

            B = len(batch_images)
            image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
            dino_features = self.dino_encoder(image_tensors)

            B = dino_features.shape[0]
            dino_encoded_features = dino_features.view(B, -1,dino_features.shape[-1])  # [B, num_view * token, dim]
            dino_encoded_features = self.dino_pro(dino_encoded_features) # B, 256, D

        with torch.autocast("cuda", dtype=torch.float32):

            start_layer = self.config.framework.layer_qformer.qformer_start_layer
            end_layer = self.config.framework.layer_qformer.qformer_end_layer
            condition_features = qwenvl_outputs.hidden_states[start_layer:end_layer]
            cat_conditions = []
            for layer_index in range(len(condition_features)):
                layer_features = condition_features[layer_index]  # [B, n_qformer_token, D]
                layer_features = torch.cat([layer_features, dino_encoded_features], dim=1)  # [B, n_qformer_token + num_view * token, D] 
                cat_conditions.append(layer_features)

            action_condition_feature = self.layer_qformer(cat_conditions) # --> [B, 64, D_action]
    
            using_cfg = cfg_scale > 1.0

            model_dtype = next(self.action_model.net.parameters()).dtype
            B = action_condition_feature.shape[0]

            # Sample random noise
            noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=action_condition_feature.device).to(model_dtype)  #[B, T, D]

            # Setup classifier-free guidance:
            if using_cfg:
                noise = torch.cat([noise, noise], 0) #[2,16,7]
                uncondition = self.action_model.net.z_embedder.uncondition # [64, 768]
                uncondition_shape = uncondition.shape
                uncondition = uncondition.unsqueeze(0)  #[1, 64, D]
                uncondition = uncondition.expand(B, uncondition_shape[0], uncondition_shape[1]) #[B, n_qformer_token, D] # 
                z = torch.cat([action_condition_feature, uncondition], 0) # [2, 64, 768] TODO check 看看 training的时候是剁手
                cfg_scale = cfg_scale
                model_kwargs = dict(z=z, cfg_scale=cfg_scale)
                sample_fn = self.action_model.net.forward_with_cfg
            else:
                model_kwargs = dict(z=action_condition_feature)
                sample_fn = self.action_model.net.forward
            
            # DDIM Sampling
            if use_ddim and num_ddim_steps is not None:
                if self.action_model.ddim_diffusion is None:
                    self.action_model.create_ddim(ddim_step=num_ddim_steps)
                samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=action_condition_feature.device,
                                                                    eta=0.0
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

    config_yaml = "InternVLA/config/lerobot_data/qwenvla_cotrain_oxe.yaml"
    cfg = OmegaConf.load(config_yaml)

    # try get model
    model = build_model_framework(cfg)
    print(model)

    # try forward model
    # can be fake sample， but here get from dataloader for simpler

    from code.dataloader.lerobot_datasets_oxe import get_vla_dataset, collate_fn

    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )
    
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        batch
        break
    
    # try get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model(batch)
