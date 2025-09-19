from collections import deque
from typing import Optional, Sequence
import os
import torch
import cv2 as cv
import numpy as np
from PIL import Image
from transforms3d.euler import euler2axangle

from InternVLA.model.framework.M1 import InternVLA_M1 as QwenpiPolicy
from eval.sim_cogact.adaptive_ensemble import AdaptiveEnsembler


class QwenpiPolicyInterfence:
    def __init__(
        self,
        saved_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
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
        unnorm_key = unnorm_key or "franka"
        action_ensemble_horizon = 2
        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

        # load model
        self.vla = QwenpiPolicy.from_pretrained(saved_model_path)
        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
        self.vla = self.vla.to("cuda").eval()

        # parameter setup
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.image_size = image_size
        self.action_scale = action_scale
        self.cfg_scale = cfg_scale
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

        # state management
        self.task_description = None
        self.image_history = deque(maxlen=0)  # not use history image
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        # action ensemble
        if action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(action_ensemble_horizon, adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None

    def reset(self, task_description: str) -> None:
        """reset policy state"""
        self.task_description = task_description
        if self.action_ensembler:
            self.action_ensembler.reset()

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(
        self, images, task_description: Optional[str] = None, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        execute one step inference
        :param image: input image (H, W, 3) uint8 format
        :param task_description: task description text
        :return: (raw action, processed action)
        """
        # reset task description
        if task_description and task_description != self.task_description:
            self.reset(task_description)

        task_description = self.align_text_input(task_description or self.task_description)
        # ensure image format correct --> here to align data format, including size, requirements and model alignment
        pil_images = self.align_visual_input(images)  # images is a list, with one element

        # model inference
        CoT_sentences, normalized_actions = self.vla.predict_action(
            images=[pil_images],  # batch size = 1
            instructions=[task_description],
            unnorm_key=self.unnorm_key,
            do_sample=False,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
        )

        # unnormalize action
        action_norm_stats = self.vla.get_action_stats(self.unnorm_key)
        raw_actions = self.vla.unnormalize_actions(
            normalized_actions=normalized_actions[0], action_norm_stats=action_norm_stats  # rm B
        )  # 16, 7 --> chunck, dim

        # action ensemble
        if self.action_ensembler and False:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        # parse raw action
        raw_action = {
            "xyz_delta": raw_actions[0][:3],
            "rotation_delta": raw_actions[0][3:6],
            "open_gripper": raw_actions[0][6:7],  # 0 is open
        }

        return raw_action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """resize image and keep RGB format"""
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    def init_infer(self, stettings):
        """initialize inference state"""
        self.stettings = stettings
        self.image_history.clear()
        self.reset(self.task_description)
        print("Policy interface initialized.")

    def align_visual_input(self, images: Sequence[np.ndarray]) -> list[Image.Image]:
        """
        align visual input format
        :param images: input image list, each image is (H, W, 3) uint8 format
        :return: PIL image list
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

    def align_text_input(self, text: str) -> str:
        """
        align text input format
        :param text: input text
        :return: text list
        """
        return text.strip()
