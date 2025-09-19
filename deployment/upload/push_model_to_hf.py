from huggingface_hub import create_repo, HfApi

# 1. create repository
hf_name = "InternRobotics/InternVLA-M1-Pretrain"
create_repo(hf_name, repo_type="model", exist_ok=True)

# 2. initialize API
api = HfApi()

# 3. upload large folder
folder_path = "/mnt/petrelfs/share/yejinhui/Models/Pretrained_models/Qwen2.5-VL-3B-Instruct_where2place_65"
# 4. use upload_large_folder to upload
api.upload_large_folder(folder_path=folder_path, repo_id=hf_name, repo_type="model")
