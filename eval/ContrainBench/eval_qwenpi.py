import re
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from llavavla.model.framework.qwenpi import QwenQFormerDiT
from llavavla.training.metrics import TrainerUtils
from llavavla.dataloader import build_dataloader
import debugpy

# Debug setup
debugpy.listen(("0.0.0.0", 10092))
print("🔍 Rank 0 waiting for debugger attach on port 10092...")
debugpy.wait_for_client()

class QWenPiModelEvaluator: # TODO 它不应该是某个模型的测试， 他应该是某个bench 为中心的绑定， 只是要求模型返回 bbox 和 action
    def __init__(self, model_path, config=None):
        """
        初始化VLA模型评估器
        
        Args:
            model_path (str): 预训练模型路径
            config: 模型配置对象
        """
        self.model = QwenQFormerDiT.from_pretrained(model_path)
        self.processor = self.model.qwen_vl_interface.processor
        self.config = config # TODO 这里应该采用传染的行config
        
        # 构建数据集和数据加载器
        self.vla_dataset, self.collate_fn = build_dataloader(self.config)
        self.dataloader = DataLoader(
            self.vla_dataset,
            batch_size=self.config.datasets.vla_data.per_device_batch_size,
            collate_fn=self.collate_fn
        )
        
        # 评估结果存储
        self.total_iou_scores = []
        self.total_action_distances = []

    @staticmethod
    def extract_json_from_string(input_string):
        """
        从字符串中提取有效的JSON部分
        
        Args:
            input_string (str): 可能包含JSON的字符串
            
        Returns:
            dict: 解析后的JSON字典，失败返回None
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                print(f"JSON解码失败: {input_string}")
                return None
        return None

    def evaluate_predictions(self, predicted_solutions, solutions, normalized_actions, actions):
        """
        评估预测结果
        
        Args:
            predicted_solutions: 预测的解决方案列表
            solutions: 真实解决方案列表
            normalized_actions: 归一化的预测动作
            actions: 真实动作
            
        Returns:
            dict: 包含IoU分数和动作距离的评估结果
        """
        batch_iou_scores = []
        batch_action_distances = []
        
        for pred_solution, gt_solution, pred_action, gt_action in zip(
            predicted_solutions, solutions, normalized_actions, actions
        ):
            try:
                # 解析预测和真实解决方案
                pred_dict = pred_solution if isinstance(pred_solution, dict) else self.extract_json_from_string(pred_solution)
                gt_dict = eval(gt_solution) if isinstance(gt_solution, str) else gt_solution
                
                # 计算边界框IoU
                pred_pick = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                
                pick_iou = box_iou(pred_pick, gt_pick).item()
                place_iou = box_iou(pred_place, gt_place).item()
                
                # 计算动作距离
                action_distance = TrainerUtils.euclidean_distance(pred_action, gt_action)
                num_elements = np.prod(pred_action.shape)
                avg_action_distance = action_distance / num_elements
                
                batch_iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})
                batch_action_distances.append(avg_action_distance)
                
            except Exception as e:
                print(f"处理预测时出错: {e}")
                batch_iou_scores.append({"pick_iou": 0.0, "place_iou": 0.0})
                batch_action_distances.append(0.0)
        
        return {
            "iou_scores": batch_iou_scores,
            "action_distances": batch_action_distances
        }

    def run_evaluation(self, max_batches=20):
        """
        运行评估循环
        
        Args:
            max_batches (int): 最大评估批次数量
        """
        dataloader_iter = iter(self.dataloader)
        # TODO 怎么想办法只测试开头结尾之类的 key frames? 这里的dataloader， 怎么获取整条轨迹为单位进行测试？
        for batch_idx in range(max_batches):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
            
            # 准备输入数据
            images = [example["image"] for example in batch]
            instructions = [example["lang"] for example in batch]
            actions = np.array([example["action"] for example in batch])
            solutions = [example["solution"] for example in batch]
            
            # 模型预测
            predicted_solutions, normalized_actions = self.model.predict_action(
                images=images,
                instructions=instructions,
                use_ddim=False,
                num_ddim_steps=20
            )
            
            # 解析预测结果
            parsed_solutions = []
            for solution in predicted_solutions:
                parsed = self.extract_json_from_string(solution)
                parsed_solutions.append(parsed if parsed is not None else {"pick": {"bbox_2d": [0,0,0,0]}, "place": {"bbox_2d": [0,0,0,0]}})
            
            # 评估当前批次
            eval_results = self.evaluate_predictions(
                parsed_solutions, solutions, normalized_actions, actions
            )
            
            # 更新总体结果
            self.total_iou_scores.extend(eval_results["iou_scores"])
            self.total_action_distances.extend(eval_results["action_distances"])
            
            # 计算并打印当前统计信息
            avg_pick_iou = np.mean([iou["pick_iou"] for iou in self.total_iou_scores])
            avg_place_iou = np.mean([iou["place_iou"] for iou in self.total_iou_scores])
            avg_action_dist = np.mean(self.total_action_distances)
            
            print(
                f"Batch {batch_idx + 1}: "
                f"Pick IoU: {avg_pick_iou:.4f}, "
                f"Place IoU: {avg_place_iou:.4f}, "
                f"Action Distance: {avg_action_dist:.4f}"
            )

        # 最终评估结果
        final_results = {
            "average_pick_iou": avg_pick_iou,
            "average_place_iou": avg_place_iou,
            "average_action_distance": avg_action_dist,
            "total_samples": len(self.total_iou_scores)
        }
        
        print("\n=== Final Evaluation Results ===")
        print(json.dumps(final_results, indent=2))
        
        return final_results

# 使用示例
from omegaconf import OmegaConf
if __name__ == "__main__":
    model_path = "/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0712_vla_v4_vlma/checkpoints/steps_14000_pytorch_model.pt"

    # Load YAML config & Convert CLI overrides to dotlist config
    config_yaml = "llavavla/conf/qwenvla_lmdb_genmanip.yaml"
    cfg = OmegaConf.load(config_yaml)
    
    evaluator = QWenPiModelEvaluator(model_path, config=cfg)
    evaluation_results = evaluator.run_evaluation(max_batches=20)