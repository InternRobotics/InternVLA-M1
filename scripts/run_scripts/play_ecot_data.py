
import os.path as osp
import pickle
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import os
import json

if os.environ.get("DEBUG", None) or 1:
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("🔍 Rank 0 waiting for debugger attach on port 5678...")
    debugpy.wait_for_client()

# 指定 JSON 文件路径
json_path = "/mnt/petrelfs/yejinhui/Projects/llavavla/playground/Datasets/embodied_features_bridge/embodied_features_bridge.json"

# 打开并读取 JSON 文件内容
with open(json_path, "r") as f:
    data = json.load(f)
# 打印前几个键值对作为预览
for key in list(data.keys())[:1]:
    sample = data[c]
    print(f"{key}: {data[key]}")


pass


# key: task name?
'/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/bridge_data_v2/deepthought_folding_table/stack_blocks/19/train/out.npy'
len(data[key])
45

episode = data[key]["0"] # episode 0?
len(data[key]["0"]["metadata"]) # 43
metadata = data[key]["0"]["metadata"]


len(data[key]["0"]["reasoning"]) # 43
reasoning = data[key]["0"]["reasoning"]

len(data[key]["0"]["features"]["bboxes"]) # 43

bboxes_in_episode = data[key]["0"]["features"]["bboxes"] # 是 224 * 224 scale 下的么？

bboxes_in_step_0 = data[key]["0"]["features"]["bboxes"][0]

# 数据好像有点 脏？
metadata["language_instruction"]
bboxes_in_step_0 # box name 有点对不上



