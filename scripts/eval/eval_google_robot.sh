cd /mnt/petrelfs/yejinhui/Projects/llavavla



MODEL_DIR=/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0_bar/0723_v6_vla_dino_32_2_need
step=30000

MODEL_PATH=${MODEL_DIR}/checkpoints/steps_${step}_pytorch_model.pt
LOG_PATH=${MODEL_DIR}/videos/steps_${step}_pytorch_model.pt/logs
mkdir -p $LOG_PATH


SCRIPT_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/eval/sim_cogact/scripts/qwenact/cogact_drawer_variant_agg.sh
# 2 * (378 / 42) = 18
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "${LOG_PATH}/drawer_variant.log" 2>&1 &

sleep 1

SCRIPT_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/eval/sim_cogact/scripts/qwenact/cogact_drawer_visual_matching.sh
# 216 / 32 = 6.75
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "${LOG_PATH}/drawer_visual_matching.log"  2>&1 &

sleep 1

SCRIPT_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/eval/sim_cogact/scripts/qwenact/cogact_move_near_variant_agg.sh

# 10
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "$LOG_PATH/move_near_variant.log" 2>&1 &

sleep 1

SCRIPT_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/eval/sim_cogact/scripts/qwenact/cogact_move_near_visual_matching.sh
# 4
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "$LOG_PATH/move_near_visual_matching.log" 2>&1 &

sleep 1

SCRIPT_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/eval/sim_cogact/scripts/qwenact/cogact_pick_coke_can_variant_agg.sh
# 33
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "$LOG_PATH/pick_coke_can_variant" 2>&1 &

sleep 1

SCRIPT_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/eval/sim_cogact/scripts/qwenact/cogact_pick_coke_can_visual_matching.sh
#  12
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "$LOG_PATH/pick_coke_can_visual_matching.log" 2>&1 &

sleep 1

SCRIPT_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/eval/sim_cogact/scripts/qwenact/cogact_put_in_drawer_variant_agg.sh

# 7
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "$LOG_PATH/put_in_drawer_variant.log" 2>&1 &

sleep 1

SCRIPT_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/eval/sim_cogact/scripts/qwenact/cogact_put_in_drawer_visual_matching.sh
# 12
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "$LOG_PATH/put_in_drawer_visual_matching.log" 2>&1 &



