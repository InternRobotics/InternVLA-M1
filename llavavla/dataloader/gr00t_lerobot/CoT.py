import os, json

def get_episode_cot(self, episode_name, frame_index, obs= "image_0", dir="/mnt/petrelfs/share/efm_p/yujunqiu/grounding/filtered_results"):

    json_path  = os.path.join(dir, f"{episode_name}_bboxes.json") # TODO 不同数据下的格式现在不一样，之后需要统一 --> 将CoT作为新的表征加入到 HF 数据集中
    
    
    # 这里数据构建到的问题太多
    # pick_name = None
    # place_name = None
    # pick_bbox = None
    # place_bbox = None
    original_key = self.lerobot_modality_meta.video[obs].original_key
    #load json
    solution_sentence = None
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            CoT_episodic = json.load(f)
            image_annotaions = CoT_episodic.get(original_key, {})

        object_list = list(image_annotaions.keys())

        solution_sentence = ""
        for obj in object_list:
            obj_bbox = image_annotaions[obj].get(f"{frame_index}", None)
            if obj_bbox is not None:
                solution_sentence += f"{obj} is at {obj_bbox}"
            
    return solution_sentence # 返回的东西可能是 none

