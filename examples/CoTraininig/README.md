
## Co-Training VLA with VLM Data Guide🚀

This guide outlines the process for integrating VLM data to co-train the VLA framework, enhancing its general visual and language understanding.

---

### 📦 1. Multi-Modal Data Preparation

The VLM data must adhere to the [QwenVL Conversations JSON Data Structure](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune).

#### Required Format:
* Each data instance is a JSON object.
* It links an **image file path** to a list of **human-GPT conversational turns**.

```json
{
    "image": "path/to/images/001.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nWhat's the main object in this picture?"
        },
        {
            "from": "gpt",
            "value": "A red apple on a wooden table"
        }
    ]
}
````

#### Data Recipe:
The primary open-source multi-modal dataset used in InternVLA-M1 is sourced from[LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)

Please download the data and reformat it according to the [QwenVL Conversations JSON Data Structure](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune).


-----

### ⚙️ 2. VLM Dataset Configuration

To add a custom VLM dataset, follow these steps:

#### 2.1 Register Dataset (Python)

Register your dataset by adding it to the `data_dict` in [qwen_data_config.py](./https://github.com/InternRobotics/InternVLA-M1/blob/711bf49498a4875b175835362d88d4462351db2f/InternVLA/dataloader/qwenvl_llavajson/qwen_data_config.py#L9).


```python
# Example Registration

SHAREGPT4V_COCO = {
    "annotation_path": f"{json_root}/sharegpt4v_coco.json",
    "data_path": f"{image_root}/",
}

data_dict = {
    "sharegpt4v_coco": SHAREGPT4V_COCO, # Use this name in the YAML config
}
```

#### 2.2 Update Training YAML

Include the VLM dataset configuration in your training YAML file (`your_train_config.yaml`).

```yaml
datasets:
  vlm_data:
    dataset_py: vlm_datasets
    dataformat: llava_json
    dataset_use: sharegpt4v_coco # Must match the name registered in 2.1
```

**Tip:** You can verify the VLM dataloader by running:

```bash
python InternVLA/dataloader/vlm_datasets.py --config_yaml your_train_config.yaml
```


-----

### 🚀 3. Co-Train VLA with VLM Data

This simultaneously trains the model on both robotics (VLA) and multi-modal (VLM) data.

  * **Script:** `InternVLA/training/train_internvla_cotrain.py`

<!-- end list -->

```bash
bash scripts/run_scripts/run_lerobot_datasets_cotrainvl.sh
```

