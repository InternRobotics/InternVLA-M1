
import os
import lmdb
import pickle
import numpy as np
import cv2
import json
from PIL import Image, ImageDraw

import random

# Configuration
root_dir = '/mnt/petrelfs/yejinhui/Projects/llavavla/playground/Datasets/vis'
image_size = (640, 480)
num_samples = 10
horizon = 10
step_length = 5
visual = True
output_image_dir = './playground/Datasets/vis/tem_images'
output_parquet = './playground/Datasets/vis/debug_with_visual.parquet'

from typing import Optional, Dict, List
import pickle




class ObjectInfoManager:
    def __init__(self):
        # 两个 JSON 文件路径
        self.mapping_files = [
            '/mnt/petrelfs/yejinhui/Projects/System2VLA/buildata/assets/uid_caption_mapping_simplified.json',
            '/mnt/petrelfs/yejinhui/Projects/System2VLA/buildata/assets/uid_caption_mapping.json'
        ]
        self.object_info_simplified = {}
        self.object_info_full = {}
        self.use_simplified = True  # 用于交替选择映射
        self.load_mappings()

    def load_mappings(self):
        """
        加载两个 JSON 文件的 UID 到名称映射。
        """
        with open(self.mapping_files[0], "r", encoding="utf-8") as f:
            self.object_info_simplified = json.load(f)
        with open(self.mapping_files[1], "r", encoding="utf-8") as f:
            self.object_info_full = json.load(f)

    def uid2name(self, uid: str) -> Optional[str]:
        """
        根据 UID 获取对象的名称，随机选择使用简化映射或完整映射。
        """
        if True:  # 随机选择 True 或 False
            name = self.object_info_simplified.get(uid, None)
        else:
            name = self.object_info_full.get(uid, None)
        return name

obj_info_mgr = ObjectInfoManager()
obj_info_mgr.load_mappings()  # 确保加载了两个映射文件


# Ensure output directory
os.makedirs(output_image_dir, exist_ok=True)

# Utility Functions
def open_lmdb_env(episode_path: str) -> tuple:
    """
    打开 LMDB 环境并加载 meta_data。
    返回： (txn, meta_data)
    txn: lmdb.Transaction in a with-block context
    meta_data: dict from meta_info.pkl
    """
    lmdb_path = os.path.join(episode_path, 'lmdb')
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False
    )
    meta_data = pickle.load(open(os.path.join(episode_path, 'meta_info.pkl'), 'rb'))
    return env, meta_data


def get_bbox_and_label_arrays(key_bbox, key_label, frame_idx: int) -> tuple:
    """
    从 txn 中读取指定帧的 loose bbox 数组与 id2label dict。
    返回： (boxes_np, labels_dict)
    boxes_np: numpy array shape (N,5) 每行 [label, xmin,ymin,xmax,ymax]
    labels_dict: dict[label_id->{class: uid}]
    """

    frame_boxes = np.stack([np.array(x).item() for x in key_bbox[frame_idx]])
    labels_list = key_label[frame_idx]
    return frame_boxes, labels_list


def compute_bbox_for_uid(frame_boxes: np.ndarray, labels_list: dict, uid: int,
                         scale_x: float=1.0, scale_y: float=1.0) -> list:
    """
    给定 frame_boxes 和对应的 labels_list，筛选出目标 uid 的 bbox 并缩放到 image_size。
    返回： [xmin, ymin, xmax, ymax]
    """
    id_to_uid = {v['class']: int(k) for k, v in labels_list.items()}
    label_id = id_to_uid[uid] # 这里会有bug， 说明 meta 中的 labels_list 没有覆盖到 拿到的 pick uid  
    mask = frame_boxes[:,0] == label_id
    x1,y1,x2,y2 = frame_boxes[mask, 1:5][0]
    name = obj_info_mgr.uid2name(str(uid))
    return [int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)], name

def get_all_objects_and_bboxes(raw_boxes, raw_obj_layouts, frame_idx: int,
                               scale_x: float = 1.0, scale_y: float =1.0,
                               obj_info_mgr = None) -> list:
    """
    获取当前帧中所有可识别物体的 UID、caption 和 bbox。
    返回格式: [{'uid': xxx, 'name': 'a red banana...', 'bbox': [x1,y1,x2,y2]}, ...]
    """

    boxes = np.stack([np.array(x).item() for x in raw_boxes[frame_idx]])
    labels_dict = raw_obj_layouts[frame_idx]

    object_list = []
    for label_id, info in labels_dict.items():
        uid = info['class']
        name = obj_info_mgr.uid2name(str(uid))
        if name is None:
            continue  # 跳过未找到物体信息的 UID

        mask = boxes[:, 0] == int(label_id)
        if not np.any(mask): continue
        x1, y1, x2, y2 = boxes[mask, 1:5][0]
        bbox = [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]

        object_list.append({'uid': uid, 'name': name.strip(), 'bbox': bbox})
    return object_list

def get_action_trajectory(all_trace_2d: list, start_idx: int,
                          horizon: int, step_length: int,
                          scale_x: float, scale_y: float) -> list:
    """
    从 all_trace_2d 中，计算未来 horizon 步的第2点投影并缩放。
    `trace2d[i]` shape (6,2)，取 index=1。
    返回 list of [x,y]
    """
    T = len(all_trace_2d)
    traj = []
    for m in range(horizon):
        idx = min(start_idx + m*step_length, T-1)
        pts = all_trace_2d[idx]
        x,y = pts[1]
        traj.append([int(x*scale_x), int(y*scale_y)])
    return traj

def get_trajectory_plan(
    all_trace_2d: list,
    start_idx: int = 0,
    end_idx: int = -1,
    horizon: int=10,
    scale_x: float = 1.0,
    scale_y: float = 1.0
) -> list:
    """
    从 start_idx 到 end_idx 之间，均匀采样 horizon 个第二投影点，并缩放到图像坐标。
    如果 horizon 大于帧数差（end_idx - start_idx + 1），会对相邻帧做线性插值以补齐采样点。

    参数:
      all_trace_2d: list of shape [(n_points, 2), ...]，每帧若有多点，则选第 2 个点
      start_idx: 起始帧索引
      end_idx: 结束帧索引（包含）
      horizon: 采样点总数
      scale_x, scale_y: 缩放因子

    返回:
      traj: list of [x, y]，长度为 horizon
    """
    T = len(all_trace_2d)
    if end_idx == -1:
        end_idx = T - 1
    if not (0 <= start_idx < T):
        raise IndexError(f"start_idx {start_idx} out of range [0, {T})")
    if not (0 <= end_idx < T):
        raise IndexError(f"end_idx {end_idx} out of range [0, {T})")
    if end_idx <= start_idx:
        raise ValueError(f"end_idx ({end_idx}) must be > start_idx ({start_idx})")
    if horizon < 2:
        raise ValueError("horizon must be at least 2 to interpolate")

    # 在 [start_idx, end_idx] 区间生成均匀浮点索引
    positions = np.linspace(start_idx, end_idx, num=horizon)

    traj = []
    for pos in positions:
        # 若 pos 正好是整数索引，直接取该帧
        if pos.is_integer():
            idx = int(pos)
            pts = all_trace_2d[idx]
            x, y = pts[1]
        else:
            # 否则对 floor 和 ceil 帧坐标做线性插值
            lo, hi = int(np.floor(pos)), int(np.ceil(pos))
            alpha = pos - lo
            pts_lo = all_trace_2d[lo][1]
            pts_hi = all_trace_2d[hi][1]
            x = (1 - alpha) * pts_lo[0] + alpha * pts_hi[0]
            y = (1 - alpha) * pts_lo[1] + alpha * pts_hi[1]

        # 缩放并取整
        traj.append([int(x * scale_x), int(y * scale_y)])

    return traj




def decode_and_resize_image(txn, key: str, image_size: tuple) -> Image.Image:
    raw_img = cv2.imdecode(np.frombuffer(pickle.loads(txn.get(key)), np.uint8), cv2.IMREAD_COLOR)
    return Image.fromarray(raw_img).resize(image_size, Image.Resampling.LANCZOS)


from PIL import ImageDraw, ImageFont
import json

def annotate_sample(sample: dict,
                    img=None,
                    current_bbox_color='red',
                    target_bbox_color='green',
                    traj_color='blue',
                    affordance_color='red',
                    bbox_width=3,
                    point_radius=3,
                    status_color='yellow',
                    instruction_color='white',
                    font_path=None,
                    font_size=20) -> Image.Image:
    """
    在 sample['image'] 上可视化当前 bbox、目标 bbox、action 轨迹 与 affordance 点，
    并在 affordance 点旁显示 current_status，在顶部显示 instruction。
    返回带标注的 PIL.Image。
    """
    sol = json.loads(sample['solution'])
    curr = sol.get('pick_bbox', None)
    tgt = sol.get('place_bbox', None) 
    traj = sol.get('future_traj', [])
    aff_pt = sol.get('curr_affordance_point', None)
    current_status = sol.get('current_obj', None)
    instruction = sample.get('language_instruction', None)
    current_griper_bbox = sol.get('current_gripper_bbox', None)
    # 准备用于绘制
    if img is None:
        img = sample['image']
    img = img.copy()
    draw = ImageDraw.Draw(img)

    # 尝试加载字体，否则用默认
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 当前 bbox
    if curr is not None:
        draw.rectangle([(curr[0], curr[1]), (curr[2], curr[3])],
                    outline=current_bbox_color, width=bbox_width)
    

    if tgt is not None:
        # 目标 bbox
        draw.rectangle([(tgt[0], tgt[1]), (tgt[2], tgt[3])],
                    outline=target_bbox_color, width=bbox_width)
    if current_griper_bbox is not None:
        # 当前抓取 bbox
        draw.rectangle([(current_griper_bbox[0], current_griper_bbox[1]), (current_griper_bbox[2], current_griper_bbox[3])],
                    outline=target_bbox_color, width=bbox_width)
    # 轨迹点
    for (x, y) in traj:
        draw.ellipse((x-point_radius, y-point_radius, x+point_radius, y+point_radius),
                     fill=traj_color)

    # 抓取时刻点
    if aff_pt is not None:
        ax, ay = aff_pt
        # 标示 affordance 点
        draw.ellipse((ax-5, ay-5, ax+5, ay+5), fill=affordance_color)

        # 如果有 current_status，就在 aff_pt 旁显示
        if current_status is not None:
            text = str(current_status)
            # 文本位置稍微偏右下
            txt_x, txt_y = ax + 8, ay + 8
            draw.text((txt_x, txt_y), text, fill=status_color, font=font)

    # 在图片顶部显示 instruction
    if instruction:
        # 在顶部留个 margin
        margin_x, margin_y = 10, 10
        # 如果 instruction 很长，可自行拆行或缩短
        draw.text((margin_x, margin_y), instruction, fill=instruction_color, font=font)

    return img

def visualize_objects_with_bboxes(sample: dict,
                                  img= None,
                                  bbox_color='orange',
                                  text_color='black',
                                  bbox_width=2,
                                  with_uid: bool = False,
                                  ) -> Image.Image:
    """
    从 sample 中可视化 all_objects 的 bbox + label。
    返回标注后的 PIL.Image。
    """
    sol = json.loads(sample['solution'])
    object_list = sol.get('all_object_list', [])
    if img == None:
        img = sample['image']
    img = img.copy()
    draw = ImageDraw.Draw(img)

    for obj in object_list:
        x1, y1, x2, y2 = obj['bbox']
        name = obj['name']
        uid = obj.get('uid', '')
        label = f"{name}" if not with_uid else f"{name} ({uid})"
        draw.rectangle([(x1, y1), (x2, y2)], outline=bbox_color, width=bbox_width)
        draw.text((x1 + 4, y1 + 4), label, fill=text_color)

    return img


def detect_affordance_point(current_index, gripper_close: list, trace_2d: list,
                            point_index: int = 1) -> Optional[Dict]:
    """
    查找 gripper 从开到闭的时刻，并返回该帧的指定关键点坐标。
    默认使用抓取点 index=2。

    返回:
        {'frame_idx': i, 'point': [x, y]} 或 None（如果没有变化）
    """
    assert len(gripper_close) == len(trace_2d)
    prev_index = max(0, current_index)
    prev = gripper_close[prev_index]
    for i in range(current_index, len(gripper_close)):
        curr = gripper_close[i]
        if prev == -1 and curr == 1:
            pt = trace_2d[i][point_index]
            return [int(pt[0]), int(pt[1])]
        prev = curr
    return None


def detect_first_change_point(
    current_index: int,
    gripper_close: List[int],
    trace_2d: List[List[List[float]]],
    point_index: int = 1
) -> Optional[Dict]:
    """
    从 current_index 开始，找到 gripper_close 列表中第一次发生值变化的时刻，
    并返回该帧的指定关键点坐标。

    参数:
        current_index: 开始检测的帧索引
        gripper_close: 手爪状态列表（例如，-1 表示张开，1 表示闭合）
        trace_2d: 每帧的 2D 关键点列表，形状为 [帧数][关键点数][2]
        point_index: 要返回的关键点在每帧列表中的索引

    返回:
        {'frame_idx': i, 'point': [x, y]} 或 None（如果到末尾都没有变化）
    """
    assert len(gripper_close) == len(trace_2d), "长度不一致，无法对应每一帧"

    prev = gripper_close[current_index]
    # 从下一个帧开始检测
    for i in range(current_index + 1, len(gripper_close)):
        curr = gripper_close[i]
        if curr != prev:
            x, y = trace_2d[i][point_index]
            return {
                'frame_idx': i,
                'point': [int(x), int(y)]
            }
        prev = curr

    # 没有检测到任何变化
    return None


def get_current_phase(frame_index: int, frame_status: Dict[str, int]) -> Optional[str]:
    """
    根据给定的帧索引，返回该帧当前执行的阶段。

    参数:
        frame_index: 当前帧的索引 i
        frame_status: 阶段字典，键为 "<trial>/<phase>"，值为该阶段开始的帧索引

    返回:
        当前阶段名称（去掉 trial 前缀），
        如果 frame_index 在最早阶段开始之前，则返回 None。
    """
    # 解析并提取 (start_frame, phase_name) 列表
    phases = []
    for key, start in frame_status.items():
        # key 形如 "0/pre_grasp"，我们只取后半部分作为阶段名
        _, phase = key.split("/", 1)
        phases.append((start, phase))
    
    # 按开始帧排序
    phases.sort(key=lambda x: x[0])
    
    current_phase: Optional[str] = None
    # 遍历各阶段，找到最大的 start <= frame_index
    for start, phase in phases:
        if frame_index >= start:
            current_phase = phase
        else:
            break
    
    return current_phase

def sample_frame_each_phase(frame_status: Dict[str, int], total_frames: int) -> Dict[str, int]:
    """
    从每个阶段中随机采样一个帧。

    参数:
        frame_status: 阶段字典，键为 "<trial>/<phase>"（如 "0/grasp"），值为该阶段开始帧索引
        total_frames: 视频或序列的总帧数

    返回:
        dict: 每个阶段对应采样到的帧索引，键为 phase 名（不含 trial 前缀）
    """
    # 解析并排序各阶段
    phases = []
    for key, start in frame_status.items():
        _, phase = key.split('/', 1)
        phases.append((start, phase))
    phases.sort(key=lambda x: x[0])

    sampled = []
    sampled.append((phases[0][1], 2))

    for idx, (start, phase) in enumerate(phases):
        # 计算该阶段的结束帧（下一个阶段起始帧 - 1，或总帧数 - 1）
        if idx + 1 < len(phases):
            end = phases[idx + 1][0] - 1
        else:
            end = total_frames - 1
        # 避免区间无效
        if end < start:
            frame = start
        else:
            frame = random.randint(start, end)
        if phase in ["pre_grasp", "post_grasp", "pre_place"]:
            sampled.append((phase, frame))
    
    return sampled


def scale_point(p, scale_x, scale_y):
    return [int(p[0] * scale_x), int(p[1] * scale_y)]

def scale_trace(trace_2d, scale_x, scale_y):
    return [np.array([scale_point(p,scale_x, scale_y) for p in frame]) for frame in trace_2d]




def get_spatial_relation_8dir(bbox1, bbox2, threshold: float = 15.0) -> str:
    x1 = (bbox1[0] + bbox1[2]) / 2
    y1 = (bbox1[1] + bbox1[3]) / 2
    x2 = (bbox2[0] + bbox2[2]) / 2
    y2 = (bbox2[1] + bbox2[3]) / 2

    dx = x2 - x1
    dy = y2 - y1

    # 忽略非常小的偏移（可选）
    horiz = ""
    vert = ""

    if abs(dx) > threshold:
        horiz = "right" if dx > 0 else "left"

    if abs(dy) > threshold:
        vert = "behind" if dy > 0 else "in front"

    # 拼接方向
    if horiz and vert:
        return f"{vert}-{horiz} of"   # e.g. "in front-right of"
    elif horiz:
        return f"to the {horiz} of"
    elif vert:
        return f"{vert} of"
    else:
        return "at the same position"


import os
import json
import random
from tqdm import tqdm
from PIL import Image

def process_single_sample_conception(sample: dict, image_name: str, image_dir: str) -> dict | None:
    """
    将一个 sample 转成 Conception 风格的记录。
    返回 None 则跳过该 sample。
    """
    sol = json.loads(sample["solution"])
    all_objs = sol.get("all_object_list", [])
    if len(all_objs) < 2:
        return None

    image_name = f"{image_name}.jpg"
    # 假设图片已保存到 image_dir/image_name
    image_path = os.path.join(image_dir, image_name)
    # Q1: 列出所有物体
    convo = [
        {"from": "human", "value": "<image>\nWhat objects are present in this image? Output the object list in JSON format."},
        {"from": "gpt",   "value": json.dumps([o["name"].strip() for o in all_objs], indent=2)}
    ]

    # Q2: 第一个随机物体在哪
    obj1 = random.choice(all_objs)
    convo += [
        {"from": "human", "value": f"Where is the '{obj1['name']}'?"},
        {"from": "gpt",   "value": json.dumps({"bbox_2d": obj1["bbox"]})}
    ]

    # Q3: 第二个随机物体的空间关系
    others = [o for o in all_objs if o != obj1]
    if not others:
        return None
    obj2 = random.choice(others)
    
    relation = get_spatial_relation_8dir(obj1["bbox"], obj2["bbox"])
    convo += [
        {"from": "human",
         "value": f"What is the spatial relationship of '{obj2['name']}' to '{obj1['name']}'?"},
        {"from": "gpt",
         "value": f"The '{obj2['name'].strip()}' is {relation} the '{obj1['name'].strip()}'."}
    ]

    return {"id": image_name, "images": [image_path], "conversations": convo}


def process_single_sample_task(sample: dict, image_name: str, image_dir: str) -> dict | None:
    """
    将一个 sample 转成 VLA-task 风格的记录。
    返回 None 则跳过该 sample。
    """
    sol = json.loads(sample["solution"])
    image_name = f"{image_name}.jpg"
    # 假设图片已保存到 image_dir/image_name
    image_path = os.path.join(image_dir, image_name)
    cur_obj = sol.get("current_obj")
    instr  = sample.get("language_instruction", "")
    cur_box = sol.get("current_bbox")
    tgt_box = sol.get("target_bbox")
    aff_pt  = sol.get("curr_affordance_point")
    fut_trj = sol.get("future_traj")

    is_grasped = aff_pt is not None

    convo = [
        # ① 操作对象
        {"from": "human",
         "value": f"<image>\nYour task is: \"{instr}\"\nAccording to the scene, which object should be operated on?"},
        {"from": "gpt", "value": f"{cur_obj}"},

        # ③ 位置信息
        {"from": "human",
         "value": f"To complete '{instr}', where is the object now? And where to place it? Reply in JSON."},
        {"from": "gpt",
         "value": json.dumps({"current_bbox": cur_box, "target_bbox": tgt_box})},

        # ⑤ 抓取状态
        {"from": "human",
         "value": f"Is the '{cur_obj}' currently grasped by the gripper?"},
        {"from": "gpt",
         "value": "Yes." if is_grasped else "No."}
    ]

    # ⑦ 抓取点
    if is_grasped:
        convo += [
            {"from": "human",
             "value": f"Where is the grasp point in the image to complete '{instr}'?"},
            {"from": "gpt", "value": json.dumps(aff_pt)}
        ]

    # ⑨ 轨迹预测（robot gripper）
    convo += [
        {"from": "human",
         "value": f"The task is: '{instr}'. Based on the current state, what is the expected motion trajectory of the robot gripper? Please respond in JSON."},
        {"from": "gpt",
         "value": json.dumps({"future_traj": fut_trj})}
    ]

    return {"id": image_name, "images": [image_path], "conversations": convo}



def transform_to_single_turn_json(samples, output_path):
    """
    将构造好的 sample 列表转换为 one-turn 对话格式，并写入 JSON 文件。
    每条记录只有两条对话：
        1. human: 带任务指令的 prompt
        2. gpt:   JSON 答案（包含 bbox、grasp、traj）
    """
    out = []
    for idx, sample in enumerate(samples):
        try:
            # 提取必要信息
            instr = sample.get("language_instruction", "").strip()
            parsed_sol = json.loads(sample.get("solution", "{}"))
            
            current_obj = parsed_sol.get("current_obj", ""),
            current_bbox = parsed_sol.get("current_bbox", None)
            target_bbox  = parsed_sol.get("target_bbox", None)
            grasp_point  = parsed_sol.get("curr_affordance_point", None)
            future_traj  = parsed_sol.get("future_traj", None)

            # 构造 prompt
            prompt = (
                "<image>\n"
                f"Your task is: \"{instr}\".\n"
                "Based on the current scene, please output in JSON format:\n"
                "- the object current need to pick,\n"
                "- the object's current bounding box,\n"
                "- the object's target area to place,\n"
                # "- the grasp point (or null if not grasped),\n"
                # "- and the future motion trajectory of the gripper."
            )

            # GPT 回答
            response = {
                "pick_object": current_obj,
                "pick_object_bbox": current_bbox,
                "place_area_bbox":  target_bbox,
                # "grasp_point":  grasp_point,
                # "future_traj":  future_traj
            }

            # 构造单轮对话样本
            out.append({
                "id": f"{sample['id']}",
                "images": [f"images/{sample['id']}.jpg"],
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": json.dumps(response)}
                ],
                "meta_info": sample["meta_info"],
            })

        except Exception as e:
            print(f"❌ Error processing sample {idx}: {e}")
            continue

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(out)} one-turn samples to {output_path}")

def concat_instruction(task_data):
    instruction = ""
    for i, subgoal in enumerate(task_data["goal"][0]):
        # lower case and remove "."

        obj1_uid = subgoal["obj1_uid"]
        obj1_name = obj_info_mgr.uid2name(obj1_uid)


        obj2_uid = subgoal["obj2_uid"]
        obj2_name = obj_info_mgr.uid2name(obj2_uid)


        position = subgoal["position"]
        position = "top" 
        if i == 0:
            instruction += (
                f"Move the {obj1_name} to the {position} of the {obj2_name}"
            )
        else:
            instruction += (
                f", and move the {obj1_name} to the {position} of the {obj2_name}"
            )
    instruction += "."
    return instruction


def check_sample(sample, min_bbox_area=0.001):
    """过滤掉无效 sample，例如 bbox 无效、轨迹为空、bbox 太小等"""
    try:
        sol = json.loads(sample['solution'])
        w,h = sample["image"].size
        min_bbox_area = (w * h) * min_bbox_area    # 640x480 是原始图像大小
        # 当前 bbox 和目标 bbox 都要是有效的 4 元素 list
        current_bbox = sol.get('pick_bbox')
        target_bbox = sol.get('place_bbox')

        if not (isinstance(current_bbox, list) and len(current_bbox) == 4):
            return False
        if not (isinstance(target_bbox, list) and len(target_bbox) == 4):
            return False

        # 计算 bbox 面积
        def bbox_area(bbox):
            x1, y1, x2, y2 = bbox
            return max(0, x2 - x1) * max(0, y2 - y1)

        if bbox_area(current_bbox) < min_bbox_area:
            return False
        if bbox_area(target_bbox) < min_bbox_area:
            return False

        return True  # 通过所有检查
    except Exception as e:
        print(f"[Filter] Sample {sample.get('id')} 过滤失败：{e}")
        return False
    

def points_to_bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    """
    给定一个 (N,2) 数组 points，每行是 [x, y]，
    返回能包住所有点的最小边界框 (x1, y1, x2, y2)。

    参数:
        points: numpy 数组，shape = (N,2)

    返回:
        (x1, y1, x2, y2): float 型坐标
    """
    if points.size == 0:
        raise ValueError("输入的 points 数组不能为空")
    xs = points[:, 0]
    ys = points[:, 1]
    x1 = xs.min()
    y1 = ys.min()
    x2 = xs.max()
    y2 = ys.max()
    return x1, y1, x2, y2


def numpy_encoder(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    # optionally: handle other types here
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


