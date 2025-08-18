from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from llavavla.model.framework.qwenpi import QwenQFormerDiT
import os, torch


import re
import json

def extract_json_from_string(input_string):
    """
    从字符串中提取有效的 JSON 部分并转换为字典。
    
    Args:
        input_string (str): 包含多余字符的字符串。
    
    Returns:
        dict: 提取并解析后的字典。
    """
    # 使用正则表达式提取 JSON 部分
    json_match = re.search(r"{.*}", input_string, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            # 转换为字典
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON 解码失败: {e}")
            return None
    else:
        print("未找到有效的 JSON 部分")
        return None

from llavavla.training.trainer_utils.metrics import TrainerUtils
from torchvision.ops import box_iou
import numpy as np

def evaluate_predictions(predicted_solutions, solutions, normalized_actions, actions):
    """
    评价预测结果，包括 IoU 和动作距离。
    
    Args:
        predicted_solutions: List[json]，预测的解决方案（JSON格式）。
        solutions: List[str]，真实的解决方案（JSON格式字符串）。
        normalized_actions: np.ndarray，预测的动作。
        actions: np.ndarray，真实的动作。

    Returns:
        dict: 包含 IoU 和动作距离的评价结果。
    """
    iou_scores = []
    action_distances = []
    for pred_solution, gt, pre_action, gt_action in zip(predicted_solutions, solutions, normalized_actions, actions):
        # pred_dict = eval(pred_solution)
        try:
            pred_dict = pred_solution
            gt_dict = eval(gt)

            # 提取 bbox
            pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
            gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
            pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
            gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

            # 计算 IoU
            pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
            place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

            # 计算动作距离
            actions = np.array(pre_action)  # 确保 actions 是 numpy 数组
            num_pots = np.prod(pre_action.shape)  # B*len*dim
            action_distance = TrainerUtils.euclidean_distance(pre_action, gt_action)
            average_action_distance = action_distance / num_pots

            # add results
            action_distances.append(average_action_distance)
            iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})
        except:
            print(f"Error processing prediction: {pred_solution} or ground truth: {gt}")
            iou_scores.append({"pick_iou": 0.0, "place_iou": 0.0})
            action_distances.append(0.0)

    return {
        "iou_scores": iou_scores,
        "action_distances": action_distances
    }

import debugpy, torch
debugpy.listen(("0.0.0.0", 10092))
print("🔍 Rank 0 waiting for debugger attach on port 10092...")
debugpy.wait_for_client()

saved_model_path = "/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0712_vla_v4_vlma/checkpoints/steps_14000_pytorch_model.pt"

qwenpi:QwenQFormerDiT = QwenQFormerDiT.from_pretrained( # a lot of Missing key(s) in state_dict:
          saved_model_path,                       # choose from ['CogACT/CogACT-Small', 'CogACT/CogACT-Base', 'CogACT/CogACT-Large'] or the local path
        )


# default: Load the model on the available device(s)

processor = qwenpi.qwen_vl_interface.processor


cfg = qwenpi.config

from llavavla.dataloader import build_dataloader
from llavavla.training.trainer_utils.metrics import TrainerUtils
import numpy as np
from torch.utils.data import DataLoader

vla_train_dataloader = build_dataloader( # 这个写在dataload.py 内部
    cfg=cfg)

# 方法2: 使用迭代器
dataset_iter = iter(vla_train_dataloader)
count = 0

total_iou_scores = []
total_avg_action_distance = []
while True and count < 20 :
    try:
        batch_samples = next(dataset_iter)
        count += 1
    except StopIteration:
        break

    examples = batch_samples
    score = 0.0 # 想办法看看证明变成batch 推理
    num_samples = len(examples)

    # @Jinhui TBD TODO 
    images = [example["image"] for example in examples]  #  TODO check 是什么
    instructions = [example["lang"] for example in examples]  # [B, str]
    actions = [example["action"] for example in examples] # action label
    solutions = [example["solution"] for example in examples]  # [B, str]
    # Predict actions using the model
    predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT( # TODO 这里有 模型方法 依赖关系, 如果你要保持trainer的独立性，这里应该怎么设计？
        images=images,
        instructions=instructions,
        use_ddim=False,
        num_ddim_steps=20)
    
    # 提取并转换
    parsed_solutions = []
    for solution in predicted_solutions:
        parsed_solution = extract_json_from_string(solution)
        parsed_solutions.append(parsed_solution)


    # 提前转换 actions 为 numpy.ndarray
    actions = np.array(actions)  # 将 actions 转换为 numpy.ndarray (B, len, dim)
    # B, Chunk, dim = actions.shape
    num_pots = np.prod(actions.shape) # B*len*dim
    # Compute the metric score
    score = TrainerUtils.euclidean_distance(normalized_actions, actions)
    average_score = score / num_pots

    # 调用评价函数
    evaluation_results = evaluate_predictions(parsed_solutions, solutions, normalized_actions, actions)

    # print avg score from evaluation_results
    iou_scores = evaluation_results["iou_scores"]
    action_distances = evaluation_results["action_distances"]
    total_iou_scores.extend(iou_scores)
    total_avg_action_distance.extend(action_distances)

    avg_pick_iou = np.mean([iou["pick_iou"] for iou in total_iou_scores])
    avg_place_iou = np.mean([iou["place_iou"] for iou in total_iou_scores])
    avg_action_distance = np.mean(total_avg_action_distance)
    print(f"Batch {count}: Average Pick IoU: {avg_pick_iou:.4f}, Average Place IoU: {avg_place_iou:.4f}, Average Action Distance: {avg_action_distance:.4f}")
