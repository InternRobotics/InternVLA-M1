"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""




from typing import Tuple
import re
import json
import numpy as np
import torch

from accelerate.logging import get_logger
logger = get_logger(__name__)

# TODO 这里或许写到trainer 内部更好？
# TODO 之类的文件是否需要重构？

# === Define Tracker Interface ===
# 

# utils/cli_parser.py

def normalize_dotlist_args(args): # 其实可以交给 OmegaConf 内部的， 但是考虑到要给用户暴露这个参数的构建过程
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        else:
            pass  # skip orphaned values
    return normalized


def build_param_lr_groups(model, cfg): # TODO 后面要和 trainer 绑定
    """
    根据 cfg.trainer.learning_rate 构建多 param group 的参数组。
    支持指定模块使用不同学习率，其余使用 base。
    
    Args:
        vla: nn.Module 模型对象
        cfg: 配置对象，要求有 cfg.trainer.learning_rate 字典

    Returns:
        List[Dict]: 可用于 torch.optim 构建 optimizer 的 param_groups
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # 默认 base 学习率

    used_params = set()
    param_groups = []

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # 尝试按 module_name 在 vla 下找到模块（支持嵌套路径）
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            params = list(module.parameters())
            param_groups.append({"params": params, "lr": lr, "name": module_name})
            used_params.update(id(p) for p in params)
        except AttributeError:
            ReferenceError(f"⚠️ 模块路径 `{module_name}` 无法在 vla 中找到")

    # 将其余未使用的参数分配 base 学习率
    other_params = [p for p in model.parameters() if id(p) not in used_params]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


import torch.distributed as dist

def only_main_process(func):
    """
    装饰器：仅在主进程（rank=0）时运行
    """
    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None  # 非主进程不执行
        return func(*args, **kwargs)
    return wrapper


from torchvision.ops import box_iou
from PIL import Image
def resize_images(images, target_size=(224, 224)):
    """
    递归调整嵌套列表中的所有图像大小。
    
    :param images: 嵌套的图像列表或单个图像。
    :param target_size: 调整后的目标大小 (width, height)。
    :return: 调整大小后的图像列表，保持原始嵌套结构。
    """
    if isinstance(images, Image.Image):  # 如果是单个 PIL 图像
        return images.resize(target_size)
    elif isinstance(images, list):  # 如果是列表，递归处理每个元素
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")

import torch.distributed as dist

class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        根据相对模块路径列表（patterns）直接冻结指定子模块，不再递归查找所有子模块名称：
          - patterns: 从 config.trainer.freeze_modules 中读取，用逗号分隔得到的“相对路径”列表
            例如 "qwen_vl_interface, action_model.net"，
            就意味着冻结 model.qwen_vl_interface 和 model.action_model.net。
        返回值：
          - model: 
        """
        frozen = []
        if freeze_modules:
            # 拆分并去除空白
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # 将“相对路径”按点拆分，例如 "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # 如果成功 get 到 module，就把它和它的所有子模块参数都 freeze
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # 如果某一级属性不存在，就跳过并打印警告
                    print(f"⚠️ 模块路径不存在，无法冻结：{path}")
                    continue

        dist.barrier()  # 分布式训练时同步
        print(f"🔒 Frozen modules (by relative path): {frozen}")
        return model
    
    @staticmethod 
    def print_trainable_parameters(model):
        """
        打印模型的总参数数量和可训练参数数量
        :param model: PyTorch 模型实例
        """
        if dist.get_rank() != 0:
            return
        print("📊 模型参数统计：")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable")
        return num_params, num_trainable_params
    
    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None):
        """
        加载 checkpoint：
        - 如果设置了 reload_modules 按路径部分加载
        - 否则 → 加载整个模型参数（覆盖 model）

        返回：
            替换，loaded_modules: 成功加载参数的模块路径列表；若全局加载则为 ["<full_model>"]
        """
        if not checkpoint_path:
            return []  
        if dist.get_rank() == 0:
            print(f"📦 正在加载 checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ 加载 checkpoint 失败: {e}")

        loaded_modules = []

        if reload_modules:  # 部分加载
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                reload_modules = path.split(".")
                module = model
                try:
                    for module_name in reload_modules:  # 逐级找到要修改的模块
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {
                        k[len(prefix):]: v
                        for k, v in checkpoint.items()
                        if k.startswith(prefix)
                    }
                    if sub_state_dict:
                        module.load_state_dict(sub_state_dict, strict=True)
                        if dist.get_rank() == 0:
                            print(f"✅ 参数已加载到模块 '{path}'")
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ checkpoint 中未找到 '{path}' 相关参数")
                except AttributeError:
                    print(f"❌ 无法找到模块路径：{path}")
        else:  # 全部加载
            try:
                model.load_state_dict(checkpoint, strict=True)
                if dist.get_rank() == 0:
                    print("✅ 已加载<full_model>模型参数")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"❌ 加载完整模型失败: {e}")
        return model
    
    @staticmethod
    def print_freeze_status(model):
        """
        打印模型中每个参数的冻结状态
        :param model: PyTorch 模型实例
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        使用 Accelerator 准备分布式训练组件
        :param accelerator: Accelerate 的实例
        :param components: 任意数量的组件（如模型、优化器、数据加载器等）
        :return: 准备好的分布式组件（与输入顺序一致）
        """
        # 使用 accelerator.prepare 方法包装组件
        prepared_components = accelerator.prepare(*components)
        return prepared_components
    @staticmethod
    def euclidean_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.linalg.norm(predicted - ground_truth)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """安全重置dataloader迭代器"""
        # 1. 更新epoch计数
        epoch_counter += 1
        
        # 2. 设置新epoch（分布式核心）
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)
        
        # 3. 创建新迭代器
        return iter(dataloader), epoch_counter
    
    @staticmethod
    def compute_grad_angle_with_stats(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> Tuple[float, float]:
        """
        计算两组梯度向量的余弦夹角（度），并统计平均夹角和方差。
        grads_a, grads_v: 与同一参数列表 interface_params 对应的梯度 Tensor 列表
        返回:
            mean_angle_deg: 平均夹角（度）
            angle_variance: 夹角方差
        """
        angle_degs = []
        
        # TODO 怎么看这个夹角才合理？
        # 分块计算每个梯度的夹角 grads_a[0].shape = 1280, 3, 14, 14
        # 梯度太多不好看？
        # grads_1 = grads_a[0][0]  # 形状为 [3, 14, 14]
        # grads_2 = grads_v[0][0]
        # grads_a = grads_1.view(-1, 3)  # 重塑为 [196, 3]
        # grads_v = grads_2.view(-1, 3)

        # lang linear
        # reshape 为 14*14, 3
        # layer
        grads_action = grads_a[0]  # 形状为 [2048, 11008]
        grads_action = grads_action[:32, :7] # 只取前7个元素, 避免高维空间cosim 失效
        grads_vl = grads_v[0]  # 形状为 [2048, 11008]
        grads_vl = grads_vl[:32, :7] # 只取前32个元素, 7 维度, 避免高维空间cosim 失效
        # PCA 在看？FVD full rank
        for g_a, g_v in zip(grads_action, grads_vl):
            dot = torch.sum(g_a * g_v)
            norm_a_sq = torch.sum(g_a * g_a)
            norm_v_sq = torch.sum(g_v * g_v)

            # 避免除零
            norm_a = torch.sqrt(norm_a_sq + 1e-16)
            norm_v = torch.sqrt(norm_v_sq + 1e-16)

            cos_sim = (dot / (norm_a * norm_v)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = angle_rad * (180.0 / torch.pi)

            angle_degs.append(angle_deg.item())

        # 计算平均夹角和方差
        angle_degs_tensor = torch.tensor(angle_degs)
        mean_angle_deg = torch.mean(angle_degs_tensor).item()
        angle_variance = torch.sqrt(torch.var(angle_degs_tensor)).item()
        # dist.barrier() # @DEBUG
        return mean_angle_deg, angle_variance

    @staticmethod
    def pcgrad_project(grads_a: list[torch.Tensor],
                    grads_v: list[torch.Tensor]
                    ) -> list[torch.Tensor]:
        """
        对第二组梯度 grads_v 应用 PCGrad 投影，抑制与 grads_a 间的负迁移
        如果两组梯度的点积 < 0，则：
            grads_v <- grads_v - (dot / ||grads_a||^2) * grads_a
        返回新的 grads_v 列表
        """
        # 先算 dot 和 ||grads_a||^2
        dot, norm_a_sq = 0.0, 0.0
        for g_a, g_v in zip(grads_a, grads_v):
            dot       += torch.sum(g_a * g_v)
            norm_a_sq += torch.sum(g_a * g_a)

        if dot < 0:
            coeff = dot / (norm_a_sq + 1e-6)
            # 投影
            grads_v = [g_v - coeff * g_a for g_a, g_v in zip(grads_a, grads_v)]

        return grads_v

    @staticmethod
    def eval_qwenpi(qwenpi, dataloader, num_batches=20):  # TODO 这个方法解耦性不够好
        """
        评估 QwenQFormerDiT 模型，计算 IoU 和动作距离。
        
        Args:
            qwenpi: QwenQFormerDiT 模型实例。
            dataloader: 数据加载器。
            num_batches: 评估的批次数量。
        
        Returns:
            dict: 包含 IoU 和动作距离的评价结果。
        """
        iou_scores = []
        action_distances = []
        count = 0

        dataset_iter = iter(dataloader)
        while count < num_batches:
            try:
                batch_samples = next(dataset_iter)
                count += 1
            except StopIteration:
                break

            # 提取数据
            images = [example["image"] for example in batch_samples]
            instructions = [example["lang"] for example in batch_samples]
            actions = [example["action"] for example in batch_samples]
            solutions = [example["solution"] for example in batch_samples]

            # 模型预测
            predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT(
                images=images,
                instructions=instructions,
                use_ddim=False,
                num_ddim_steps=20
            )

            # 提取并转换预测结果
            parsed_solutions = []
            for solution in predicted_solutions:
                parsed_solution = TrainerUtils.extract_json_from_string(solution)
                parsed_solutions.append(parsed_solution)

            # 计算 IoU
            for pred_dict, gt_dict in zip(parsed_solutions, solutions):
                pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

                pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
                place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

                iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})

            # 计算动作距离
            actions = np.array(actions)  # 转换为 numpy 数组
            num_pots = np.prod(actions.shape)  # B*len*dim
            action_distance = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_action_distance = action_distance / num_pots
            action_distances.append(average_action_distance)

        # 汇总结果
        avg_action_distance = np.mean(action_distances)
        return {
            "iou_scores": iou_scores,
            "average_action_distance": avg_action_distance
        }

    @staticmethod
    def extract_json_from_string(input_string): # TODO 这个方法解耦性不够好
        """
        从字符串中提取有效的 JSON 部分并转换为字典。
        
        Args:
            input_string (str): 包含多余字符的字符串。
        
        Returns:
            dict: 提取并解析后的字典。
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON 解码失败: {e}")
                return None
        else:
            print("未找到有效的 JSON 部分")
            return None
import os

def is_main_process(): # TODO 要变成一个修饰函数， 但是是否可以像 if 你要修饰？ 就是修饰每个逻辑？
    rank = int(os.environ.get("RANK", 0))  # 如果未设置 RANK，则默认为 0
    return rank == 0