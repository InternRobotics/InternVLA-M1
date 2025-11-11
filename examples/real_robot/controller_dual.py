"""
InternVLA-M1 Dual Frequency Controller for Real Robot Deployment

This module provides a dual-frequency control system for robotic manipulation
using the InternVLA-M1 model with async planning capabilities.
"""

import argparse
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger
from sanic import Sanic, response

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from InternVLA.model.framework.M1 import InternVLA_M1
from InternVLA.model.framework.share_tools import read_mode_config
from examples.real_robot.adaptive_ensemble import AdaptiveEnsembler
from examples.real_robot.tool import get_camera_data, formulate_input

app = Sanic("InferenceServer")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def get_parser():
    """Create argument parser for controller configuration."""
    parser = argparse.ArgumentParser(description="InternVLA-M1 Dual Frequency Controller")
    
    # Model configuration
    parser.add_argument("--saved_model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", 
                        help="Path to the saved model")
    parser.add_argument("--policy_setup", type=str, default="real",
                        help="Policy setup type (use 'real' for real robot deployment)")
    parser.add_argument("--use_bf16", action="store_true", 
                        help="Use bfloat16 precision")
    
    # Action ensemble configuration
    parser.add_argument("--action_ensemble", action="store_true", default=True,
                        help="Use action ensemble")
    parser.add_argument("--adaptive_ensemble_alpha", type=float, default=0.1,
                        help="Adaptive ensemble alpha")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                        help="CFG scale for inference")
    
    # Action configuration
    parser.add_argument("--action_type", type=str, default="delta_ee_pose",
                        help="Action type: delta_ee_pose, abs_ee_pose, delta_qpos, abs_qpos")
    
    # Dual frequency mode parameters
    parser.add_argument("--dual_freq_mode", action="store_true", default=True,
                        help="Enable dual frequency mode (sys2 + sys1)")
    parser.add_argument("--planning_freq", type=float, default=2.0,
                        help="Planning frequency in seconds (total time for one sys2 planning cycle: inference + sleep)")
    
    return parser


class M1Inference:
    """
    InternVLA-M1 inference class with dual frequency control.
    
    Supports async planning (sys2) and action generation (sys1) at different frequencies.
    """
    
    def __init__(
        self,
        saved_model_path: str = 'Qwen/Qwen2.5-VL-3B-Instruct',
        unnorm_key: str = None,
        policy_setup: str = "real",
        horizon: int = 0,
        action_ensemble_horizon: int = None,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        use_bf16: bool = False,
        action_ensemble: bool = True,
        adaptive_ensemble_alpha: float = 0.1,
        # Action type parameter
        action_type: str = "delta_ee_pose",
        # Dual frequency mode parameters
        dual_freq_mode: bool = True,
        planning_freq: float = 2.0,  # High-level planning cycle time in seconds (inference + sleep)
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Set up policy setup defaults for real robot deployment
        if policy_setup == "real":
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 16
            self.sticky_gripper_num_repeat = 1
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported. Use 'real' for real robot deployment."
            )
        
        self.policy_setup = policy_setup
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        
        # Load the model
        self.vla = InternVLA_M1.from_pretrained(saved_model_path)
        self.unnorm_key = list(self.vla.norm_stats.keys())[0]
        
        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
        self.vla = self.vla.to("cuda").eval()
        self.cfg_scale = cfg_scale

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        
        # Get action normalization stats
        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=saved_model_path)

        # Store action type
        self.action_type = action_type
        
        # Dual frequency mode parameters
        self.dual_freq_mode = dual_freq_mode
        self.planning_freq = planning_freq
        self.step_counter = 0
        
        # Async planning related (simplified)
        self.planning_thread = None
        self.current_subtask = None
        self.planning_lock = threading.RLock()
        self.is_planning_active = False
        self.should_stop_planning = False  # Control planning thread stop
        
        # Task state management (simplified)
        self.high_level_instruction = None  # High-level instruction
        self.has_initial_task = False       # Whether initial task is obtained
        self.planning_active = False        # Whether planning is activated
        self.latest_images = None           # Store latest images for planning
        
        # Sys2 style prompt template
        self.task_prompt_think = (
            "You are a robotic arm that follows human instructions and communicates naturally with humans when needed.\n"
            "Given the instruction and the observation images, generate a robotic plan in the following format:\n"
            "- response: Provide a natural, human-friendly reply. "
            "Output Format: <response>Your reply to the human</response>\n"
            "- reason: Describe the scene and explain the reasoning behind the planned action. "
            "Output Format: <think>Reasoning details here</think>\n"
            "- next_action: Specify the robotic arm's next action in a clear command form. "
            "Output Format: <action>Action command here</action>\n"
            "Instruction: {Instruction}"
        )
        
        # Start async planning thread
        if self.dual_freq_mode:
            self._start_async_planning_thread()

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str = None) -> None:
        """Reset the inference state and optionally set new task description."""
        if task_description is not None:
            self.task_description = task_description
            self.high_level_instruction = task_description
            self.has_initial_task = True
            
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        
        # Reset dual frequency mode related state (simplified)
        self.step_counter = 0
        with self.planning_lock:
            # If initial task exists and dual frequency mode is enabled, activate planning
            if self.dual_freq_mode and self.has_initial_task and not self.planning_active:
                self.planning_active = True
                # Start async planning thread (if not already started)
                if self.planning_thread is None or not self.planning_thread.is_alive():
                    self._start_async_planning_thread()


    def _start_async_planning_thread(self):
        """Start async planning thread."""
        if self.planning_thread is None or not self.planning_thread.is_alive():
            self.should_stop_planning = False
            self.planning_thread = threading.Thread(
                target=self._async_planning_worker, 
                daemon=True,
                name="AsyncPlanningThread"
            )
            self.planning_thread.start()

    def _async_planning_worker(self):
        """Continuous async planning worker thread."""
        while not self.should_stop_planning:
            try:
                # Check if there are high-level instructions and latest images
                with self.planning_lock:
                    if (self.planning_active and 
                        self.high_level_instruction and 
                        self.latest_images is not None):
                        
                        instruction = self.high_level_instruction
                        images = self.latest_images.copy()  # Copy current images
                        self.is_planning_active = True
                    else:
                        self.is_planning_active = False
                        time.sleep(0.1)  # Wait for task activation
                        continue
                
                # Execute planning
                start_time = time.time()
                planning_result = self._execute_planning_task(images, instruction)
                end_time = time.time()
                
                # Directly update current subtask
                new_subtask = self.extract_action_from_planning(planning_result)
                with self.planning_lock:
                    if new_subtask and new_subtask.strip():
                        self.current_subtask = new_subtask.strip()
                
                # Wait for a period after planning completion before next planning
                time.sleep(max(0, self.planning_freq - (end_time - start_time)))
                
            except Exception as e:
                logger.error(f"Async planning task execution error: {e}")
                time.sleep(1.0)  # Wait 1 second after error before continuing

    def _execute_planning_task(self, images: list[Image.Image], instruction: str) -> str:
        """Execute specific planning task."""
        try:
            with self.planning_lock:
                self.is_planning_active = True
            
            # Execute planning inference
            planning_result = self.generate_planning_text(images, instruction)
            return planning_result
            
        finally:
            with self.planning_lock:
                self.is_planning_active = False

    def _get_current_instruction(self):
        """Get the current instruction to use (simplified version)."""
        with self.planning_lock:
            # Prioritize subtask, otherwise use high-level instruction
            if self.current_subtask and self.current_subtask.strip():
                return self.current_subtask.strip(), "subtask"
            elif self.high_level_instruction:
                return self.high_level_instruction, "high_level"
            else:
                return self.task_description, "original"


    def generate_planning_text(self, images: list[Image.Image], instruction: str) -> str:
        """
        Use QwenVL for high-level planning and generate text response (sys2 functionality).
        """
        # Build conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},  # obs_camera
                    {"type": "image", "image": images[1]},  # realsense
                    {
                        "type": "text",
                        "text": self.task_prompt_think.replace("{Instruction}", instruction),
                    },
                ],
            }
        ]
        
        # Use QwenVL interface for text generation
        try:
            # Build inputs
            qwen_inputs = self.vla.qwen_vl_interface.build_qwenvl_inputs(
                images=[images], 
                instructions=[self.task_prompt_think.replace("{Instruction}", instruction)]
            )
            
            # Use generate method for text generation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated_output = self.vla.qwen_vl_interface.model.generate(
                    input_ids=qwen_inputs.input_ids,
                    attention_mask=qwen_inputs.attention_mask,
                    pixel_values=qwen_inputs.pixel_values,
                    image_grid_thw=qwen_inputs.image_grid_thw,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.vla.qwen_vl_interface.processor.tokenizer.pad_token_id,
                )
            
            # Decode generated text
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(qwen_inputs.input_ids, generated_output)
            ]
            generated_text = self.vla.qwen_vl_interface.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Planning text generation failed: {e}")
            # Fallback to simple instruction
            return f"<response>I'll help you with that task.</response><think>I can see the scene and will proceed with the task.</think><action>{instruction}</action>"

    def extract_action_from_planning(self, planning_text: str) -> str:
        """
        Extract specific action instruction from planning text.
        """
        try:
            # Extract content within <action> tags
            if "<action>" in planning_text and "</action>" in planning_text:
                action_start = planning_text.find("<action>") + 8
                action_end = planning_text.find("</action>")
                action = planning_text[action_start:action_end].strip()
                return action
            else:
                # If no tags found, return original instruction
                return planning_text
        except Exception as e:
            logger.error(f"Failed to extract action from planning: {e}")
            return planning_text

    def forward(self, obs_dict, language_instruction="", timestep=0):
        """
        Forward method with async dual frequency support (sys2 + sys1).
        """
        # Increment step counter
        self.step_counter += 1
        
        # Set task description if provided
        if language_instruction and language_instruction != self.task_description:
            self.task_description = language_instruction

        # Extract images from obs_dict and apply cropping
        image_camera = obs_dict["obs_camera"]["color_image"]
        image_realsense = obs_dict["realsense"]["color_image"]
        
        # Ensure images are numpy arrays
        if not isinstance(image_camera, np.ndarray):
            image_camera = np.array(image_camera, dtype=np.uint8)
        if not isinstance(image_realsense, np.ndarray):
            image_realsense = np.array(image_realsense, dtype=np.uint8)

        # Convert to PIL Images for planning
        pil_images = [Image.fromarray(image_camera), Image.fromarray(image_realsense)]
        
        # Update high-level instruction (if new instruction provided)
        if language_instruction and language_instruction != self.high_level_instruction:
            with self.planning_lock:
                self.high_level_instruction = language_instruction
        
        # Update latest images for planning
        with self.planning_lock:
            self.latest_images = pil_images
        
        # Async dual frequency mode processing (simplified)
        if self.dual_freq_mode and self.has_initial_task:
            # Get latest instruction for action generation (directly use current subtask or high-level instruction)
            current_instruction, instruction_source = self._get_current_instruction()
        else:
            # Non-dual frequency mode or no initial task, directly use original instruction
            current_instruction = self.task_description
            instruction_source = "original"
        
        # Convert to list of numpy arrays for action generation
        images = [image_camera, image_realsense]
        
        # Call the step method for action generation
        eepose_delta, gripper_action = self.step(images, current_instruction)
        
        # For delta_ee_pose, return complete 6D eepose delta + gripper
        if self.action_type == "delta_ee_pose":
            # eepose_delta should contain [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]
            target_eepose_delta = eepose_delta[:6]  # xyz + rpy (6D)
            target_gripper = gripper_action   # gripper (1D)
        elif self.action_type == "abs_ee_pose":
            # eepose_abs should contain [abs_x, abs_y, abs_z, abs_roll, abs_pitch, abs_yaw]
            target_eepose_delta = eepose_delta[:6]  # xyz + rpy (6D)
            target_gripper = gripper_action   # gripper (1D)
        else:
            # Processing for other action_type
            target_eepose_delta = eepose_delta[:6]
            target_gripper = gripper_action
        
        is_terminal = -1.0
        return target_eepose_delta, target_gripper, is_terminal

    def step(
        self,
        images: list[np.ndarray] | np.ndarray,
        task_description: str = None, 
        *args,
        **kwargs
    ) -> tuple[np.ndarray, float]:
        """
        Step method for action prediction using InternVLA-M1.
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        if isinstance(images, np.ndarray):
            images = [images]
        assert all(image.dtype == np.uint8 for image in images)
        
        for image in images:
            self._add_image_to_history(self._resize_image(image))
        images = [Image.fromarray(image) for image in images]

        normalized_actions = self.vla.predict_action(
            batch_images=[images], 
            instructions=[self.task_description],
            unnorm_key=self.unnorm_key,
            do_sample=False, 
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
        )['normalized_actions'][0]

        raw_actions = self.unnormalize_actions(
            normalized_actions=normalized_actions, 
            action_norm_stats=self.action_norm_stats
        )

        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
            action = raw_actions[0]
        else:
            action = raw_actions[0]
        
        eepose_action = action[:6]
        gripper_action = action[6] * 2 - 1  
        
        return eepose_action, gripper_action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size with type checking."""
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint8)
        image = cv2.resize(image, tuple(self.image_size), interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Unnormalize actions using the action normalization statistics.
        
        Args:
            normalized_actions: Normalized actions in range [-1, 1]
            action_norm_stats: Dictionary containing 'q01', 'q99', and optionally 'mask'
            
        Returns:
            Unnormalized actions
        """
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Get action normalization statistics from the model checkpoint.
        
        Args:
            unnorm_key: Key to access the normalization statistics
            policy_ckpt_path: Path to the policy checkpoint
            
        Returns:
            Dictionary containing action normalization statistics
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        unnorm_key = InternVLA_M1._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    def shutdown(self):
        """Gracefully shutdown async planning thread."""
        if self.dual_freq_mode and self.planning_thread and self.planning_thread.is_alive():
            self.should_stop_planning = True
            try:
                self.planning_thread.join(timeout=5.0)
            except Exception as e:
                logger.error(f"Error shutting down planning thread: {e}")

    def __del__(self):
        """Destructor to ensure resource cleanup."""
        try:
            self.shutdown()
        except:
            pass


def generate_action(data, agent, obs_type):
    """Generate robot action based on observation and instruction."""
    global timestamp
    obs = {}
    obs["robot"] = {"ee_pose_state": np.array(data["current_pose"])}
    images, depth = get_camera_data()
    
    # Ensure image data are numpy arrays
    image_realsense = images[0] if isinstance(images[0], np.ndarray) else np.array(images[0], dtype=np.uint8)
    image_camera = images[1] if isinstance(images[1], np.ndarray) else np.array(images[1], dtype=np.uint8)
    
    obs["obs_camera"] = {"color_image": image_camera}
    obs["realsense"] = {"color_image": image_realsense}
    
    goal = data["instruction"]
    timestep = data["timestep"]
    reset = data["reset"]

    # Save images to relative path
    save_path = os.path.join("eval_camera", timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    Image.fromarray(image_camera).save(os.path.join(save_path, f"obs_camera_{timestamp}_{timestep:04d}.jpg"))
    Image.fromarray(image_realsense).save(os.path.join(save_path, f"realsense_{timestamp}_{timestep:04d}.jpg"))
        
    # Simplified task management: continuous planning after first instruction
    if reset or not agent.has_initial_task:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent.reset(goal)
    elif goal and goal != agent.high_level_instruction:
        # Subsequent requests only update high-level instruction, no reset
        with agent.planning_lock:
            agent.high_level_instruction = goal
    
    output, gripper, _ = agent.forward(obs, goal, timestep)
    result = np.array(output)
    return result, gripper


def process_data(data, agent):
    """Process input data and generate robot action."""
    try:
        processed_eepose_delta, gripper = generate_action(data, agent, "obs_camera")
        if processed_eepose_delta is None:
            return {"message": "No action generated!"}
        # Return 6D pose + gripper (-1 or 1)
        eepose_action = processed_eepose_delta.tolist() + [gripper]
        return {"eepose_action": eepose_action}
    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        return {"error": str(e)}


# Initialize agent with parsed arguments
def create_agent():
    """Create and configure the InternVLA-M1 agent."""
    args = get_parser().parse_args()
    return M1Inference(
        saved_model_path=args.saved_model_path,
        policy_setup=args.policy_setup,
        use_bf16=args.use_bf16,
        action_ensemble=args.action_ensemble,
        adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
        cfg_scale=args.cfg_scale,
        # Action type
        action_type=args.action_type,
        # Dual frequency mode parameters
        dual_freq_mode=args.dual_freq_mode,
        planning_freq=args.planning_freq,
    )


agent = create_agent()


@app.post("/infer")
async def infer(request):
    """Main inference endpoint for robot control."""
    data = request.json
    
    obs_abs_ee_pose = formulate_input(data['data'])
    data["current_pose"] = obs_abs_ee_pose
    actions = process_data(data, agent)
    
    # Build response
    response_data = {}
    
    if "error" in actions:
        return response.json({"error": actions["error"]})
    
    # Extract 6D pose and gripper (-1 or 1)
    delta_eepose = actions['eepose_action'][:6]  # xyz + rpy (6D)
    gripper = actions['eepose_action'][6]  # -1 (open) or 1 (close)
    
    response_data["target_eepose"] = (gripper, delta_eepose)
    # If dual frequency mode, add simplified planning information
    if agent.dual_freq_mode:
        with agent.planning_lock:
            response_data["current_subtask"] = agent.current_subtask
            response_data["high_level_instruction"] = agent.high_level_instruction
            response_data["is_planning_active"] = agent.is_planning_active
        
        response_data["step_counter"] = agent.step_counter
        response_data["planning_freq"] = agent.planning_freq
        response_data["has_initial_task"] = agent.has_initial_task
        response_data["planning_active"] = agent.planning_active
        
        # Planning thread status
        response_data["planning_thread_alive"] = (
            agent.planning_thread is not None and agent.planning_thread.is_alive()
        )
    
    return response.json(response_data)


if __name__ == "__main__":
    # Set single process mode to avoid multi-process startup conflicts
    app.run(host="0.0.0.0", port=25553, workers=1, single_process=True)

