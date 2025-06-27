import re

from pathlib import Path
# 这里和 rlds 要放到不同文件夹， 避免互相混淆
system2_root="/mnt/petrelfs/yejinhui/Projects/System2VLA"
json_root = f"{system2_root}/playground/Datasets/LLaVA-OneVision-Data/decoders/llava_format"
image_root = f"{system2_root}/playground/Datasets/LLaVA-OneVision-Data/decoders/visualData"
# 新增数据集定义
MAPQA_MATHV360K = {
    "annotation_path": f"{json_root}/MapQA_MathV360K.json",
    "data_path": f"{image_root}/",
}

AOKVQA_CCAULDRON_LLAVA = {
    "annotation_path": f"{json_root}/aokvqa_cauldron_llava_format.json",
    "data_path": f"{image_root}/",
}

SHAREGPT4V_COCO = {
    "annotation_path": f"{json_root}/sharegpt4v_coco.json",
    "data_path": f"{image_root}/",
}

SHAREGPT4V_KNOWLEDGE = {
    "annotation_path": f"{json_root}/sharegpt4v_knowledge.json",
    "data_path": f"{image_root}/",
}

SHAREGPT4V_LLAVA = {
    "annotation_path": f"{json_root}/sharegpt4v_llava.json",
    "data_path": f"{image_root}/",
}

SHAREGPT4V_SAM = {
    "annotation_path": f"{json_root}/sharegpt4v_sam.json",
    "data_path": f"{image_root}/",
}

VISUALWEBINSTRUCT_FILTERED = {
    "annotation_path": f"{json_root}/VisualWebInstruct_filtered.json",
    "data_path": f"{image_root}/",
}

MAGPIE_PRO_L3_80B_MT = {
    "annotation_path": f"{json_root}/magpie_pro_l3_80b_mt.json",
    "data_path": "",  # 该数据集无 image
}

MAGPIE_PRO_L3_80B_ST = {
    "annotation_path": f"{json_root}/magpie_pro_l3_80b_st.json",
    "data_path": "",  # 该数据集无 image
}

VISUAL7W_CCAULDRON_LLAVA = {
    "annotation_path": f"{json_root}/visual7w_cauldron_llava_format.json",
    "data_path": f"{image_root}/",
}

VISUALMRC_CCAULDRON = {
    "annotation_path": f"{json_root}/visualmrc_cauldron.json",
    "data_path": f"{image_root}/",
}

VSR_CCAULDRON_LLAVA = {
    "annotation_path": f"{json_root}/vsr_cauldron_llava_format.json",
    "data_path": f"{image_root}/",
}

# Genmanip objcet in container Dataset

genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v3/object_container"
genmanip_image_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v3/object_container"
GENMANIP_conception_train = {
    "annotation_path": f"{genmanip_json_root}/genmanip_conception_train.json",
    "data_path": f"{genmanip_image_root}/",
}

GENMANIP_vla_task_dropturn_train = {
    "annotation_path": f"{genmanip_json_root}/vla_task_dropturn_train.json",
    "data_path": f"{genmanip_image_root}/",
}

# ECoT
GENMANIP_vla_task_train = {
    "annotation_path": f"{genmanip_json_root}/vla_task_train.json",
    "data_path": f"{genmanip_image_root}/",
}

# final task train
# /fs-computility/efm/yejinhui/Projects/System2VLA/playground/Datasets/Genmanip_vlm_v2/object_container/vla_task_one_eval_onlypick.json
GENMANIP_vla_task_one_train = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_train.json",
    "data_path": f"{genmanip_image_root}/",
}

GENMANIP_vla_task_onlytgt_train = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_train_onlytgt.json",
    "data_path": f"{genmanip_image_root}/",
}

GENMANIP_vla_task_1by1_train = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_train_1by1.json",
    "data_path": f"{genmanip_image_root}/",
}

# Kitchen More Train
genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v3/Kitchen/all_all_more"
genmanip_image_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v3/Kitchen/all_all_more"

GENMANIP_vla_onetask_Kichen_More_train = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_train.json",
    "data_path": f"{genmanip_image_root}",
}

GENMANIP_vla_onlytgt_Kichen_More_train = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_train_onlytgt.json",
    "data_path": f"{genmanip_image_root}",
}

GENMANIP_vla_1by1_Kichen_More_train = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_train_1by1.json",
    "data_path": f"{genmanip_image_root}",
}

# Kitchen Eval

# vla_task_one_eval
genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v3/Kitchen/all_all"
genmanip_image_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v3/Kitchen/all_all"

GENMANIP_vla_onetask_Kichen = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_eval.json",
    "data_path": f"{genmanip_image_root}",
}

GENMANIP_vla_onlytgt_Kichen = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_eval_onlytgt.json",
    "data_path": f"{genmanip_image_root}",
}





# 将所有数据集注册到 data_dict
data_dict = {
    # … 已有数据集 …
    "MapQA_MathV360K": MAPQA_MATHV360K,
    "aokvqa_cauldron_llava_format": AOKVQA_CCAULDRON_LLAVA,
    "sharegpt4v_coco": SHAREGPT4V_COCO,
    "sharegpt4v_knowledge": SHAREGPT4V_KNOWLEDGE,
    "sharegpt4v_llava": SHAREGPT4V_LLAVA,
    "sharegpt4v_sam": SHAREGPT4V_SAM,
    "VisualWebInstruct_filtered": VISUALWEBINSTRUCT_FILTERED,
    "visual7w_cauldron_llava_format": VISUAL7W_CCAULDRON_LLAVA,
    "visualmrc_cauldron": VISUALMRC_CCAULDRON,
    "vsr_cauldron_llava_format": VSR_CCAULDRON_LLAVA,

    # 仿真数据
    "GENMANIP_conception_train": GENMANIP_conception_train,
    "GENMANIP_vla_task_dropturn_train": GENMANIP_vla_task_dropturn_train,
    "GENMANIP_vla_task_one_train": GENMANIP_vla_task_one_train,
    "GENMANIP_vla_task_train": GENMANIP_vla_task_train,
    "GENMANIP_vla_task_onlytgt_train": GENMANIP_vla_task_onlytgt_train,
    "GENMANIP_vla_task_1by1_train": GENMANIP_vla_task_1by1_train,
    
    # downstream bench
    "GENMANIP_vla_onetask_Kichen": GENMANIP_vla_onetask_Kichen,
    "GENMANIP_vla_onlytgt_Kichen": GENMANIP_vla_onlytgt_Kichen,
    # Kitchen More  Train
    "GENMANIP_vla_onetask_Kichen_More_train": GENMANIP_vla_onetask_Kichen_More_train,
    "GENMANIP_vla_onlytgt_Kichen_More_train": GENMANIP_vla_onlytgt_Kichen_More_train,
    "GENMANIP_vla_1by1_Kichen_More_train": GENMANIP_vla_1by1_Kichen_More_train,
    
    "all": None
}

    # "magpie_pro_l3_80b_mt": MAGPIE_PRO_L3_80B_MT,
    # "magpie_pro_l3_80b_st": MAGPIE_PRO_L3_80B_ST,



def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    if dataset_names == ["all"]:
        dataset_names= list(data_dict.keys())
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list

# Kitchen Eval - Banana Plate
genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/Kitchen/banana_plate"
genmanip_image_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/Kitchen/banana_plate"

GENMANIP_vla_onetask_Banana = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_train.json",
    "data_path": f"{genmanip_image_root}",
}

GENMANIP_vla_onlytgt_Banana = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_eval_onlytgt.json",
    "data_path": f"{genmanip_image_root}",
}

# Kitchen Eval - Lemon Plate
genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/Kitchen/lemon_plate"
genmanip_image_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/Kitchen/lemon_plate"

GENMANIP_vla_onetask_Lemon = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_train_rewritten.json",
    "data_path": f"{genmanip_image_root}",
}

GENMANIP_vla_onlytgt_Lemon = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_eval_onlytgt.json",
    "data_path": f"{genmanip_image_root}",
}

# Kitchen Eval - Sandwich Plate
genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/Kitchen/sandwich_plate"
genmanip_image_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/Kitchen/sandwich_plate"

GENMANIP_vla_onetask_Sandwich = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_train_rewritten.json",
    "data_path": f"{genmanip_image_root}",
}

GENMANIP_vla_onlytgt_Sandwich = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_eval_onlytgt.json",
    "data_path": f"{genmanip_image_root}",
}




# Kitchen Eval - All All
genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/Kitchen/all_all"
genmanip_image_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/Kitchen/all_all"

GENMANIP_vla_onetask_AllAll = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_eval.json",
    "data_path": f"{genmanip_image_root}",
}

GENMANIP_vla_onlytgt_AllAll = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_eval_onlytgt.json",
    "data_path": f"{genmanip_image_root}",
}

# Kitchen Eval - All All More
genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/Kitchen/all_all_more"
genmanip_image_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/Kitchen/all_all_more"

GENMANIP_vla_onetask_AllAllMore = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_train.json",
    "data_path": f"{genmanip_image_root}",
}

GENMANIP_vla_onlytgt_AllAllMore = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_eval_onlytgt.json",
    "data_path": f"{genmanip_image_root}",
}

# Genmanip20
genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/Genmanip20"
genmanip_image_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/Genmanip20"

GENMANIP_vla_onetask_Genmanip20 = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_eval.json",
    "data_path": f"{genmanip_image_root}",
}

data_dict.update({"GENMANIP_vla_onetask_Genmanip20": GENMANIP_vla_onetask_Genmanip20})  

# Object Container
genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/object_container"
genmanip_image_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/object_container"

GENMANIP_vla_onetask_ObjectContainer = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_eval.json",
    "data_path": f"{genmanip_image_root}",
}


GENMANIP_vla_onlytgt_ObjectContainer = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_eval_onlytgt.json",
    "data_path": f"{genmanip_image_root}",
}

# 将这些新数据集注册到 data_dict
data_dict.update({
    "GENMANIP_vla_onetask_Banana": GENMANIP_vla_onetask_Banana,
    "GENMANIP_vla_onlytgt_Banana": GENMANIP_vla_onlytgt_Banana,
    "GENMANIP_vla_onetask_Lemon": GENMANIP_vla_onetask_Lemon,
    "GENMANIP_vla_onlytgt_Lemon": GENMANIP_vla_onlytgt_Lemon,
    "GENMANIP_vla_onetask_Sandwich": GENMANIP_vla_onetask_Sandwich,
    "GENMANIP_vla_onlytgt_Sandwich": GENMANIP_vla_onlytgt_Sandwich,
})

# 将这些新数据集注册到 data_dict
data_dict.update({
    "GENMANIP_vla_onetask_AllAll": GENMANIP_vla_onetask_AllAll,
    "GENMANIP_vla_onlytgt_AllAll": GENMANIP_vla_onlytgt_AllAll,
    "GENMANIP_vla_onetask_AllAllMore": GENMANIP_vla_onetask_AllAllMore,
    "GENMANIP_vla_onlytgt_AllAllMore": GENMANIP_vla_onlytgt_AllAllMore,
    "GENMANIP_vla_onetask_ObjectContainer": GENMANIP_vla_onetask_ObjectContainer,
    "GENMANIP_vla_onlytgt_ObjectContainer": GENMANIP_vla_onlytgt_ObjectContainer,
})

# Object Object
genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/object_object"
genmanip_image_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v4/object_object"

GENMANIP_vla_onetask_ObjectObject = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_train.json",
    "data_path": f"{genmanip_image_root}",
}

GENMANIP_vla_onlytgt_ObjectObject = {
    "annotation_path": f"{genmanip_json_root}/vla_task_one_eval.json",
    "data_path": f"{genmanip_image_root}",
}

GENMANIP_vla_task_ObjectObject = {
    "annotation_path": f"{genmanip_json_root}/vla_task_train.json",
    "data_path": f"{genmanip_image_root}",
}

GENMANIP_vla_task_dropturn_ObjectObject = {
    "annotation_path": f"{genmanip_json_root}/vla_task_dropturn_train.json",
    "data_path": f"{genmanip_image_root}",
}

# 将这些新数据集注册到 data_dict
data_dict.update({
    "GENMANIP_vla_onetask_ObjectObject": GENMANIP_vla_onetask_ObjectObject,
    "GENMANIP_vla_onlytgt_ObjectObject": GENMANIP_vla_onlytgt_ObjectObject,
    "GENMANIP_vla_task_ObjectObject": GENMANIP_vla_task_ObjectObject,
    "GENMANIP_vla_task_dropturn_ObjectObject": GENMANIP_vla_task_dropturn_ObjectObject,
})




# V5

# 注册新的 Genmanip_vlm_v5 数据集

genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v5"

# Object-container
# /fs-computility/efm/yejinhui/Projects/System2VLA/playground/Datasets/Genmanip_vlm_v5/object_container
genmanip_object_object_root = f"{genmanip_json_root}/object_container"
GENMANIP_v5_object_container = {
    "annotation_path": f"{genmanip_object_object_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_object_object_root}",
}

genmanip_object_object_root = f"{genmanip_json_root}/object_object"
GENMANIP_v5_object_object = {
    "annotation_path": f"{genmanip_object_object_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_object_object_root}",
}

# 5*5 
genmanip_kitchen_5plus5 = f"{genmanip_json_root}/Kitchen/5x5"
GENMANIP_v5_kitchen_5plus5 = {
    "annotation_path": f"{genmanip_kitchen_5plus5}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_5plus5}",
}

# 5*5 more
genmanip_kitchen_5plus5_more = f"{genmanip_json_root}/Kitchen/5x5_more"
GENMANIP_v5_kitchen_5plus5_more = {
    "annotation_path": f"{genmanip_kitchen_5plus5_more}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_5plus5_more}",
}


# Kitchen - Apple Plate
genmanip_kitchen_apple_root = f"{genmanip_json_root}/Kitchen/apple_plate"
GENMANIP_v5_kitchen_apple = {
    "annotation_path": f"{genmanip_kitchen_apple_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_apple_root}",
}

# Kitchen - Banana Plate
genmanip_kitchen_banana_root = f"{genmanip_json_root}/Kitchen/banana_plate"
GENMANIP_v5_kitchen_banana = {
    "annotation_path": f"{genmanip_kitchen_banana_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_banana_root}",
}

# Kitchen - Lemon Plate
genmanip_kitchen_lemon_root = f"{genmanip_json_root}/Kitchen/lemon_plate"
GENMANIP_v5_kitchen_lemon = {
    "annotation_path": f"{genmanip_kitchen_lemon_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_lemon_root}",
}

# Kitchen - Sandwich Plate
genmanip_kitchen_sandwich_root = f"{genmanip_json_root}/Kitchen/sandwich_plate"
GENMANIP_v5_kitchen_sandwich = {
    "annotation_path": f"{genmanip_kitchen_sandwich_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_sandwich_root}",
}

# ——注册到 data_dict 中
data_dict.update({
    "GENMANIP_v5_object_object": GENMANIP_v5_object_object,
    "GENMANIP_v5_object_container": GENMANIP_v5_object_container,
    "GENMANIP_v5_kitchen_5plus5": GENMANIP_v5_kitchen_5plus5,
    "GENMANIP_v5_kitchen_5plus5_more": GENMANIP_v5_kitchen_5plus5_more,
    "GENMANIP_v5_kitchen_apple": GENMANIP_v5_kitchen_apple,
    "GENMANIP_v5_kitchen_banana": GENMANIP_v5_kitchen_banana,
    "GENMANIP_v5_kitchen_lemon": GENMANIP_v5_kitchen_lemon,
    "GENMANIP_v5_kitchen_sandwich": GENMANIP_v5_kitchen_sandwich,
})



# V6

# 注册新的 Genmanip_vlm_v5 数据集

genmanip_json_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v6"

# Object-container
# /fs-computility/efm/yejinhui/Projects/System2VLA/playground/Datasets/Genmanip_vlm_v5/object_container
genmanip_object_object_root = f"{genmanip_json_root}/object_container"
GENMANIP_v5_object_container = {
    "annotation_path": f"{genmanip_object_object_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_object_object_root}",
}

genmanip_object_object_root = f"{genmanip_json_root}/object_object"
GENMANIP_v5_object_object = {
    "annotation_path": f"{genmanip_object_object_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_object_object_root}",
}

# 5*5 
genmanip_kitchen_5plus5 = f"{genmanip_json_root}/Kitchen/5x5"
GENMANIP_v5_kitchen_5plus5 = {
    "annotation_path": f"{genmanip_kitchen_5plus5}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_5plus5}",
}

# 5*5 more
genmanip_kitchen_5plus5_more = f"{genmanip_json_root}/Kitchen/5x5_more"
GENMANIP_v5_kitchen_5plus5_more = {
    "annotation_path": f"{genmanip_kitchen_5plus5_more}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_5plus5_more}",
}


# Kitchen - Apple Plate
genmanip_kitchen_apple_root = f"{genmanip_json_root}/Kitchen/apple_plate"
GENMANIP_v5_kitchen_apple = {
    "annotation_path": f"{genmanip_kitchen_apple_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_apple_root}",
}

# Kitchen - Banana Plate
genmanip_kitchen_banana_root = f"{genmanip_json_root}/Kitchen/banana_plate"
GENMANIP_v5_kitchen_banana = {
    "annotation_path": f"{genmanip_kitchen_banana_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_banana_root}",
}

# Kitchen - Lemon Plate
genmanip_kitchen_lemon_root = f"{genmanip_json_root}/Kitchen/lemon_plate"
GENMANIP_v5_kitchen_lemon = {
    "annotation_path": f"{genmanip_kitchen_lemon_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_lemon_root}",
}

# Kitchen - Sandwich Plate
genmanip_kitchen_sandwich_root = f"{genmanip_json_root}/Kitchen/sandwich_plate"
GENMANIP_v5_kitchen_sandwich = {
    "annotation_path": f"{genmanip_kitchen_sandwich_root}/Manipu_CoT.json",
    "data_path": f"{genmanip_kitchen_sandwich_root}",
}

# ——注册到 data_dict 中
data_dict.update({
    "GENMANIP_v5_object_object": GENMANIP_v5_object_object,
    "GENMANIP_v5_object_container": GENMANIP_v5_object_container,
    "GENMANIP_v5_kitchen_5plus5": GENMANIP_v5_kitchen_5plus5,
    "GENMANIP_v5_kitchen_5plus5_more": GENMANIP_v5_kitchen_5plus5_more,
    "GENMANIP_v5_kitchen_apple": GENMANIP_v5_kitchen_apple,
    "GENMANIP_v5_kitchen_banana": GENMANIP_v5_kitchen_banana,
    "GENMANIP_v5_kitchen_lemon": GENMANIP_v5_kitchen_lemon,
    "GENMANIP_v5_kitchen_sandwich": GENMANIP_v5_kitchen_sandwich,
})



# Grounding data

llava_format_root = "/mnt/petrelfs/share/efm_p/zhuyangkun/sys2_data/coco/coco_internvl3/qwen_224_format_minp_3136_maxp_12845056"
data_root = "/mnt/petrelfs/share/efm_p/zhuyangkun/sys2_data/coco"
# Define llava_format datasets
asv2_conversation_en = {
    "annotation_path": f"{llava_format_root}/asv2_conversation_en.jsonl",
    "data_path": f"{data_root}",  # Images may be referenced inside jsonl
}

asv2_detailed_description_en = {
    "annotation_path": f"{llava_format_root}/asv2_detailed_description_en.jsonl",
    "data_path": f"{data_root}",
}

asv2_region_captioning_en = {
    "annotation_path": f"{llava_format_root}/asv2_region_captioning_en.jsonl",
    "data_path": f"{data_root}",
}

coco_internvl_longcap_en = {
    "annotation_path": f"{llava_format_root}/coco_internvl_longcap_en.jsonl",
    "data_path": f"{data_root}",
}

coco_karpathy_train_567_en = {
    "annotation_path": f"{llava_format_root}/coco_karpathy_train_567_en.jsonl",
    "data_path": f"{data_root}",
}

coco_negative_gpt4o_en = {
    "annotation_path": f"{llava_format_root}/coco_negative_gpt4o_en.jsonl",
    "data_path": f"{data_root}",
}

coco_poetry_zh = {
    "annotation_path": f"{llava_format_root}/coco_poetry_zh.jsonl",
    "data_path": f"{data_root}",
}

coco_rem_en_zh = {
    "annotation_path": f"{llava_format_root}/coco_rem_en_zh.jsonl",
    "data_path": f"{data_root}",
}

cocorem_exist_yorn_en = {
    "annotation_path": f"{llava_format_root}/cocorem_exist_yorn_en.jsonl",
    "data_path": f"{data_root}",
}

cocotextv2_en = {
    "annotation_path": f"{llava_format_root}/cocotextv2_en.jsonl",
    "data_path": f"{data_root}",
}

cocotextv2_gpt4o_en = {
    "annotation_path": f"{llava_format_root}/cocotextv2_gpt4o_en.jsonl",
    "data_path": f"{data_root}",
}

okvqa_en = {
    "annotation_path": f"{llava_format_root}/okvqa_en.jsonl",
    "data_path": f"{data_root}"
}

refcoco_grounding_aug_en = {
    "annotation_path": f"{llava_format_root}/refcoco_grounding_aug_en.jsonl",
    "data_path": f"{data_root}",
}

refcoco_grounding_en = {
    "annotation_path": f"{llava_format_root}/refcoco_grounding_en.jsonl",
    "data_path": f"{data_root}",
}

tallyqa_coco_en = {
    "annotation_path": f"{llava_format_root}/tallyqa_coco_en.jsonl",
    "data_path": f"{data_root}",
}

toloka_grounding_aug_en = {
    "annotation_path": f"{llava_format_root}/toloka_grounding_aug_en.jsonl",
    "data_path": f"{data_root}",
}

vqav2_en = {
    "annotation_path": f"{llava_format_root}/vqav2_en.jsonl",
    "data_path": f"{data_root}",
}

vsr_en = {
    "annotation_path": f"{llava_format_root}/vsr_en.jsonl",
    "data_path": f"{data_root}",
}

# —— 注册到 data_dict 中
data_dict.update({
    "asv2_conversation_en": asv2_conversation_en,
    "asv2_detailed_description_en": asv2_detailed_description_en,
    "asv2_region_captioning_en": asv2_region_captioning_en,
    "coco_internvl_longcap_en": coco_internvl_longcap_en,
    "coco_karpathy_train_567_en": coco_karpathy_train_567_en,
    "coco_negative_gpt4o_en": coco_negative_gpt4o_en,
    "coco_poetry_zh": coco_poetry_zh,
    "coco_rem_en_zh": coco_rem_en_zh,
    "cocorem_exist_yorn_en": cocorem_exist_yorn_en,
    "cocotextv2_en": cocotextv2_en,
    "cocotextv2_gpt4o_en": cocotextv2_gpt4o_en,
    "okvqa_en": okvqa_en,
    "refcoco_grounding_aug_en": refcoco_grounding_aug_en,
    "refcoco_grounding_en": refcoco_grounding_en,
    "tallyqa_coco_en": tallyqa_coco_en,
    "toloka_grounding_aug_en": toloka_grounding_aug_en,
    "vqav2_en": vqav2_en,
    "vsr_en": vsr_en,
})


# V7 long horizon
genmanip_r2s_task2_root = f"{system2_root}/playground/Datasets/Genmanip_vlm_v7/r2s_task2"

GENMANIP_v7_r2s_task2_train = {
    "annotation_path": f"{genmanip_r2s_task2_root}/Manipu_CoT_train.json",
    "data_path": genmanip_r2s_task2_root,
}

GENMANIP_v7_r2s_task2_eval = {
    "annotation_path": f"{genmanip_r2s_task2_root}/Manipu_CoT_eval.json",
    "data_path": genmanip_r2s_task2_root,
}

data_dict.update({
    "GENMANIP_v7_r2s_task2_train": GENMANIP_v7_r2s_task2_train,
    "GENMANIP_v7_r2s_task2_eval": GENMANIP_v7_r2s_task2_eval,
})
if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
