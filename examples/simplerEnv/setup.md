> To reproduce the training results, you can use the following steps:

 ## 📦 Data Preparation
 To prepare the training data, follow the NVIDIA/Isaac-GR00T dataset instructions:
 1. Clone or download the dataset from: https://github.com/NVIDIA/Isaac-GR00T/tree/main/examples/SimplerEnv
 2. Create a symbolic link to the dataset directory:
    ```bash
    ln -s [path_to_downloaded_dataset] playground/Datasets/OXE_LEROBOT
    ```
 3. Verify the dataset structure:
    ```bash
    tree -L 1 playground/Datasets/OXE_LEROBOT/
    ```
    ```
    playground/Datasets/OXE_LEROBOT/
    ├── bridge_orig_1.0.0_lerobot
    │   ├── data
    │   ├── meta
    │   └── videos
    └── fractal20220817_data_0.1.0_lerobot
    ```

 ## 🚀 Model Training
 The fine-tuning script supports multiple configurations for different simulation environments. Use the appropriate command based on your target dataset:
 ### 1. Bridge Dataset Training
 ```bash
 python scripts/gr00t_finetune.py \
     --dataset-path /tmp/bridge_orig_lerobot/ \
     --data_config examples.SimplerEnv.custom_data_config:BridgeDataConfig \
     --num-gpus 8 \
     --batch-size 64 \
     --output-dir /tmp/bridge-checkpoints \
     --max-steps 60000 \
     --video-backend torchvision_av
 ```
 ### 2. Fractal Dataset Training
 ```bash
 python scripts/gr00t_finetune.py \
     --dataset-path /tmp/fractal20220817_data_lerobot/ \
     --data_config examples.SimplerEnv.custom_data_config:FractalDataConfig \
     --num-gpus 8 \
     --batch-size 128 \
     --output-dir /tmp/fractal-checkpoints/ \
     --max-steps 60000 \
     --video-backend torchvision_av
 ```
 ## 🎯 Model Evaluation
 Evaluation is performed using the [SimplerEnv repository](https://github.com/youliangtan/SimplerEnv/tree/main).
 ### Step 1: Start the Inference Server
 ```bash
 python scripts/inference_service.py \
     --model-path youliangtan/gr00t-n1.5-bridge-posttrain/ \
     --server \
     --data_config examples.SimplerEnv.custom_data_config:BridgeDataConfig \
     --denoising-steps 8 \
     --port 5555 \
     --embodiment-tag new_embodiment
 ```
 
 ### Step 2: Run Evaluation
 ```bash
 python eval_simpler.py --env widowx_spoon_on_towel --groot_port 5555
 ```
