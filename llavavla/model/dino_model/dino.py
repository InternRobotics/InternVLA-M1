

from collections import OrderedDict
import os

from concurrent.futures import ThreadPoolExecutor
import torch

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from torchvision import transforms
def apply_transform(view, transform):
    return transform(view)

from llavavla.model.dino_model.dino_transforms import make_classification_train_transform

class DINOv2BackBone(nn.Module):
    def __init__(self, backone_name="dinov2_vits14", output_channels=1024) -> None:
        super().__init__()
        try:
            self.body = torch.hub.load("facebookresearch/dinov2", backone_name)
        except:
            import traceback

            traceback.print_exc()
            print(f"Failed to load dinov2 from torch hub, loading from local")
            TORCH_HOME = os.environ.get("TORCH_HOME", "~/.cache/torch/")
            weights_path = os.path.expanduser(
                f"{TORCH_HOME}/hub/checkpoints/{backone_name}_pretrain.pth"
            )

            code_path = os.path.expanduser(
                f"{TORCH_HOME}/hub/facebookresearch_dinov2_main"
            )

            self.body = torch.hub.load(
                code_path, backone_name, source="local", pretrained=False
            )

            state_dict = torch.load(weights_path)
            self.body.load_state_dict(state_dict)
        if backone_name == "dinov2_vits14":
            self.num_channels = 384
        elif backone_name == "dinov2_vitb14":
            self.num_channels = 768
        elif backone_name == "dinov2_vitl14":
            self.num_channels = 1024
        elif backone_name == "dinov2_vitg14":
            self.num_channels = 1408
        else:
            raise NotImplementedError(f"DINOv2 backbone {backone_name} not implemented")
        self.dino_transform = transforms.Compose([
            transforms.Resize(224),  # 调整尺寸 #@DEBUG 发现动态resize会导致问题
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # self.dino_transform = make_classification_train_transform() ##DEBUG --> 不应该做 动态的resize?
    # @torch.no_grad()
    def forward(self, tensor):
        xs = self.body.forward_features(tensor)["x_norm_patchtokens"]

        return xs # B*views, token, dim
    
    
    def prepare_dino_input(self, img_list):
        # img_list: is a list of [PIL], each representing multi views of the same example.
        # refer to https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py
        # 定义完整的预处理（包括归一化）

        # 使用线程池并行处理每个视图
        with ThreadPoolExecutor() as executor:
            image_tensors = torch.stack([
                torch.stack(list(executor.map(lambda view: apply_transform(view, self.dino_transform), views)))
                for views in img_list
            ])


        # 将张量移动到 DINO 编码器的设备
        B, num_view, C, H, W = image_tensors.shape
        image_tensors = image_tensors.view(B * num_view, C, H, W)
        device = next(self.parameters()).device
        image_tensors = image_tensors.to(device)
        
        return image_tensors

def get_dino_model(backone_name="dinov2_vits14") -> DINOv2BackBone:

    return DINOv2BackBone(backone_name)


if __name__ == "__main__":
    dino = DINOv2BackBone()
    pass