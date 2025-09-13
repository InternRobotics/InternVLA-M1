from transformers import AutoModel, AutoProcessor
# load the model and processor
import torch
import torch
from transformers import AutoModel, AutoProcessor

# 先查看可用的文件
from huggingface_hub import list_repo_files
files = list_repo_files("IPEC-COMMUNITY/EO-1-3B")
python_files = [f for f in files if f.endswith('.py')]
print("Python文件:", python_files)


# processor = AutoProcessor.from_pretrained("IPEC-COMMUNITY/EO-1-3B", trust_remote_code=True)
# model = AutoModel.from_pretrained(
#   "IPEC-COMMUNITY/EO-1-3B", 
#   trust_remote_code=True, 
#   torch_dtype=torch.bfloat16
# ).eval().cuda()

# # prepare the model input
# batch = {
#     "observation.images.image": [img], # PIL.Image
#     "observation.images.wrist_image": [wrist_img],
#     "observation.state": [state],
#     "task": ["You are a helpful physical agent equipped with both reasoning and robotic control. \
#       You see the Tic-Tac-Toe board, think strategically, act logically, and block threats."]
# }

# # generate multimodal outputs
# output = processor.generate(model, batch)
# text = output.text
# actions = output.action.numpy()

