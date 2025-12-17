

your_ckpt=./playground/Pretrained_models/InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt
python deployment/model_server/server_policy_M1.py \
    --ckpt_path ${your_ckpt} \
    --port 10093 \
    --use_bf16