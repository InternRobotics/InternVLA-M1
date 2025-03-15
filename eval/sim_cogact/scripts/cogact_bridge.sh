# switch to SimplerEnv 
# we would run evaluation under SimplerEnv repo
# 我在docker container 内部。然后我只能在里面，不能出来，现在我遇到一个问题。
# 请你帮我尝试解决。注意，每次回答前都需要总结问题，总结我们已经尝试了什么，结论是什么，接下来需要确认的问题是什么。cmd 是什么

your_path_to_simpler=/root/envs/SimplerEnv
# ln -s ./sim_cogact ${your_path_to_simpler}/simpler_env/policies -r
cd $your_path_to_simpler
conda activate simpler_env
# for headless server
Xvfb :99 -screen 0 1024x768x24 &
# 设置Xvfb显示号（假设已启动Xvfb :99）
export DISPLAY=:99
# 3. 验证OpenGL渲染能力（应显示"GLX"和软件渲染信息）
glxinfo | grep -E "OpenGL|GLX"

# # 强制使用软件渲染
# export LIBGL_ALWAYS_SOFTWARE=1
# export GALLIUM_DRIVER=llvmpipe


# 强制Sapien使用CPU模式
# export SAPIEN_USE_GPU=0

# 修复XDG_RUNTIME_DIR警告（临时方案）
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

# export LIBGL_ALWAYS_SOFTWARE=1
# export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
# export LIBGL_ALWAYS_SOFTWARE=1

# Personal Key TODO make it as read file
export HF_TOKEN=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
export WANDB_API_KEY=943ecb8d26fc2b3879cbc2d667414974906aebb9
MODEL_PATH=/fs-computility/efm/yejinhui/Projects/CogACT/playground/Pretrained_models/CogACT-Base/checkpoints/CogACT-Base.pt



gpu_id=0
policy_model=cogact
ckpt_path=${MODEL_PATH} # CogACT/CogACT-Base CogACT/CogACT-Large CogACT/CogACT-Small

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028


CUDA_VISIBLE_DEVICES=${gpu_id} 
xvfb-run -a python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;
