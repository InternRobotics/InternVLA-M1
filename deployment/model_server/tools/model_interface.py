from collections import deque
from typing import Optional, Sequence
import os
import torch
import cv2 as cv
import numpy as np
from PIL import Image
from transforms3d.euler import euler2axangle

# 根据实际项目结构调整导入路径
from InternVLA.model.framework.M1 import InternVLA_M1 as QwenpiPolicy
from eval.sim_cogact.adaptive_ensemble import AdaptiveEnsembler


# TODO @Jinhui 之前的 PolicyInterfence 是写到 simpler 本地的， 所以它同 时包括 sim 和 model 的耦合对齐， 现在要解耦， 通过 socket 来完成
class QwenpiPolicyInterfence: # 不同 model 就应该有一个 自己独立定义的 interface, --> 通过端口的映射来 对齐keys
    def __init__( # @TODO 这里的测试时的参数由谁来控制？ --> TODO 将模型 非 framework相关的 内容移出去
        self,
        saved_model_path: str = 'Qwen/Qwen2.5-VL-3B-Instruct',
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        use_bf16: bool = False,
        action_ensemble: bool = False,
        adaptive_ensemble_alpha: float = 0.1,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.ckpt_name = saved_model_path
        unnorm_key = unnorm_key or "franka" # TODO 这些其实应该是 robot control的配置 --> 并不是，其实是中间策略问题 --> 需要由那边传输一个inital config
        action_ensemble_horizon = 2
        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        
        # 加载模型
        self.vla = QwenpiPolicy.from_pretrained(saved_model_path)
        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
        self.vla = self.vla.to("cuda").eval()

        # 参数设置
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.image_size = image_size
        self.action_scale = action_scale
        self.cfg_scale = cfg_scale
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        
        # 状态管理
        self.task_description = None
        self.image_history = deque(maxlen=0)  # 不使用历史图像
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        
        # 动作集成
        if action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                action_ensemble_horizon, adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None

    def reset(self, task_description: str) -> None:
        """重置策略状态"""
        self.task_description = task_description
        if self.action_ensembler:
            self.action_ensembler.reset()
        
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step( # 这个写给不够好
        self, 
        images, 
        task_description: Optional[str] = None,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        执行一步推理
        :param image: 输入图像 (H, W, 3) uint8格式
        :param task_description: 任务描述文本
        :return: (原始动作, 处理后的动作)
        """
        # 重置任务描述
        if task_description and task_description != self.task_description:
            self.reset(task_description) # 其实不应该这里来
        
        task_description = self.align_text_input(task_description or self.task_description)
        # 确保图像格式正确 --> 这里要对齐数据格式， 包括大小， 要求 和模型对齐
        pil_images = self.align_visual_input(images)  # images 是一个 list, 里面只有一个元素

        # 模型推理 # with CoT 的方案还需要仔细看看 train 和 infer 的时候的 结束符号怎么处理
        CoT_sentences, normalized_actions = self.vla.predict_action( # predict_action_withCoT 里面不能够在做 input 处理相关的事情了
            images=[pil_images],  # batch size = 1
            instructions=[task_description],
            unnorm_key=self.unnorm_key,
            do_sample=False,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
        )
        
        # 反归一化动作
        action_norm_stats = self.vla.get_action_stats(self.unnorm_key)
        raw_actions = self.vla.unnormalize_actions(
            normalized_actions=normalized_actions[0], #rm B
            action_norm_stats=action_norm_stats
        ) # 16, 7 --> chunck, dim
        
        # 动作集成
        if self.action_ensembler and False: # TODO  why False? @BUG? --> 这里做了 ensembler 么
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
        
        # 解析原始动作
        raw_action = {
            "xyz_delta": raw_actions[0][:3], # TODO 写法很奇怪
            "rotation_delta": raw_actions[0][3:6],
            "open_gripper": raw_actions[0][6:7], # 0 is open
        } # 这里其实和模型的 modality 部分有关，关于这个的Meta 定义应该是要有个 processor的
        
        return raw_action


    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """调整图像大小并保持RGB格式"""
        return cv.resize(
            image, 
            tuple(self.image_size), 
            interpolation=cv.INTER_AREA
        )
    
    def init_infer(self, stettings):
        """初始化推理状态"""
        self.stettings = stettings
        self.image_history.clear()
        self.reset(self.task_description)
        print("Policy interface initialized.")
    def align_visual_input( self, images: Sequence[np.ndarray]) -> list[Image.Image]:
        """
        对齐视觉输入格式
        :param images: 输入图像列表，每个图像为 (H, W, 3) uint8格式
        :return: PIL图像列表
        """
        aligned_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = self._resize_image(img)
            elif isinstance(img, Image.Image):
                img = img.resize(self.image_size, Image.ANTIALIAS)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            aligned_images.append(Image.fromarray(img))
        return aligned_images
    def align_text_input(self, text:str) ->str:
        """
        对齐文本输入格式
        :param texts: 输入文本列表
        :return: 文本列表
        """
        return text.strip()
    