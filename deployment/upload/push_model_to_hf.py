from huggingface_hub import create_repo, HfApi

# 1. 创建仓库
create_repo("StarVLA/NeoQwenPi", repo_type="model", exist_ok=True)

# 2. 初始化 API
api = HfApi()

# 3. 上传大文件夹
folder_path="/mnt/petrelfs/share/yejinhui/Models/Pretrained_models/Qwen2.5-VL-3B-Instruct_where2place_65"
# 4. 使用 upload_large_folder 上传
api.upload_large_folder(
    folder_path=folder_path,
    repo_id="StarVLA/NeoQwen",
    repo_type="model"
)
