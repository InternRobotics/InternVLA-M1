# Train & Deploy InternVLA-M1 on Real Robots

To quickly train InternVLA-M1 on your data, we recommend using the [GR00T dataloader](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#1-data-format--loading) and [Huggingface Lerobot Data](https://github.com/huggingface/lerobot) format to load your real-world data.

## Custom Real Data Configuration

**Prerequisites**: You should have built Lerobot format datasets

Follow the instructions below to customize your data configuration, embodiment tags, and training recipes:

### Step 0: Create Data Configuration
Write a configuration and register a `robot_config` with `ROBOT_TYPE_CONFIG_MAP` in `InternVLA/dataloader/gr00t_lerobot/data_config.py` based on your data keys and normalization methods.

### Step 1: Associate Robot Config with Embodiment Tag
In `InternVLA/dataloader/gr00t_lerobot/embodiment_tags.py`, associate the `robot_config` with `EmbodimentTag` to enable merging statistics of the same type of embodiment.

### Step 2: Define Dataset Mixtures
Define the dataset collections to be used in `InternVLA/dataloader/gr00t_lerobot/mixtures.py`. For example, define a mixture using two datasets:

```python
"custom_dataset_2": [
    # (dataset name, dataset weight, dataset config name)
    ("custom_dataset_name_1", 1.0, "custom_robot_config"),
    ("custom_dataset_name_2", 1.0, "custom_robot_config"),
],
```

### Step 3: Data Loading
Finally, you can test data loading by running `InternVLA/dataloader/lerobot_datasets.py` through configuration.

```python
from InternVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
from torch.utils.data import DataLoader
from pathlib import Path

vla_dataset_cfg = cfg.datasets.vla_data

data_root_dir = vla_dataset_cfg.data_root_dir
data_mix = vla_dataset_cfg.data_mix

vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

vla_train_dataloader = DataLoader(
    vla_dataset,
    batch_size=cfg.datasets.vla_data.per_device_batch_size,
    collate_fn=collate_fn,
    num_workers=8,
    # shuffle=True
)

if dist.get_rank() == 0: 
    output_dir = Path(cfg.output_dir)
    vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

# Test data loading
sample_data = vla_train_dataloader[10]
```

## Demo Data for Testing
We have prepared simulation demo data for testing purposes.
[Configuration details to be added]

## Fine-Tuning InternVLA-M1 on Your Own Data

### Step 0: Configure Training Parameters
Configure training-related parameters in `InternVLA/config/training/qwenvla_cotrain_custom.yaml`, and modify `data_mix` to your previously configured training recipe:

```yaml
datasets:
  ...
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: playground/Datasets/YOUR_DATASET_PATH
    data_mix: YOUR_DATASET_NAME
  ...
```

### Step 1: Run Fine-tuning Scripts
You can run the following scripts to fine-tune the model with your own data:

```bash
# Fine-tune the model with only your action data
accelerate launch --config_file InternVLA/config/deepseeds/deepspeed_zero2.yaml   --num_processes 8 InternVLA/training/train_internvla.py --config_yaml InternVLA/config/training/internvla_cotrain_custom.yaml

# Fine-tune the model with your action and vision-language data
accelerate launch --config_file InternVLA/config/deepseeds/deepspeed_zero2.yaml   --num_processes 8 InternVLA/training/train_internvla_cotrain.py --config_yaml InternVLA/config/training/internvla_cotrain_custom.yaml
```

Hardware Requirements

We use two A100 80GB GPUs for model fine-tuning. Other devices (such as RTX 4090) may also work but will require longer convergence time. We recommend using our default configuration parameters, though you can adjust the batch size to fit your GPU memory for optimal performance.

### Simulation Data Demo
We also provide simulation data demo playground/demo_data/sim_pick_place as training samples. You can use the following command to quickly train the model:
```
accelerate launch --config_file InternVLA/config/deepseeds/deepspeed_zero2.yaml   --num_processes 8 InternVLA/training/train_internvla.py   --config_yaml InternVLA/config/training/internvla_cotrain_sim_demo.yaml 
```

## Deploy and Inference
Coming soon