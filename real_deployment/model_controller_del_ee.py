import numpy as np
import time
import os
import torch
import pynput
from datetime import datetime
from loguru import logger
from scipy.spatial.transform import Rotation as R
import numpy as np


from embodieddeploy.controller.franka_pandapy_impedance import FrankaController
from embodieddeploy.sensor.camera.realsense import MultiRealSenseCamera
from real_deployment.utils import adjust_translation_along_quaternion, RT_to_tran_quaternion, set_seed, trans_quant_to_RT
from real_deployment.controller import ModelController

import signal
import atexit

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# ... existing imports ...



status = "ok"
want_to_pause = False
def on_press(key):
    global status
    global want_to_pause
    if key == pynput.keyboard.Key.space:
        status = "pause" if status == "ok" else "ok"
        want_to_pause = not want_to_pause
    elif str(key) == "p":
        status = "pause" if status == "ok" else "ok"
        want_to_pause = not want_to_pause
    elif key == pynput.keyboard.Key.esc:
        status = "end"
    elif str(key) == "'r'":
        status = "reset"
    return






def main(model_controller: ModelController):
    global status
    
    # global settings
    keyboard_listener = pynput.keyboard.Listener(on_press=on_press)
    keyboard_listener.start()
    set_seed(1)
    print("Current working directory:", os.getcwd())
    # visualizer = ImageVisualizer("Robot Camera View")
    # visualizer.start_visualization()



    # controller
    real_robot = FrankaController(hostname="172.16.0.2", fps=15)
    for _ in range(2):
        real_robot.get_obs()
        time.sleep(1)
    logger.info("controller init done")



    # camera
    image_width = 640
    image_height = 480
    image_fps = 60
    camera = MultiRealSenseCamera(image_width=image_width, image_height=image_height, fps=image_fps)
    for _ in range(10):
        _ = camera.undistorted_rgb()
        time.sleep(0.1)
    logger.info("camera init done")


    time.sleep(5)









    # ================= Finish Loading =================
    last_ee = real_robot.get_obs()["panda_hand_pose"]
    last_trans, last_ori = RT_to_tran_quaternion(last_ee)


    print("Robot warm up done")
    with torch.inference_mode():
        try:
            while status != "end":
                t = 0
                try:
                    while status != "end":
                        time1 = time.time()

                        # Get Input
                        lang = "Place the cucumber on the plate"
                        rgb_images = camera.undistorted_rgb()
                        # visualizer.update_images(rgb_images)

                        # Predict
                        # action = model_controller.infer_debug(t) # TODO: for debug
                        # action = model_controller.infer(rgb_images, lang)
                        action = model_controller.infer_debug_debug(t, rgb_images, lang)
                        del_trans, del_ori, gripper = action[:3], action[3:7], action[7:]

                        # Update
                        cur_trans = last_trans + del_trans
                        cur_ori = last_ori + del_ori # TODO: 取决于训练时计算 del 的方式
                        cur_ee = trans_quant_to_RT(cur_trans, cur_ori)


                        gripper = 1 # TODO: retrain gripper
                        target_eepose = [gripper, cur_ee]

                        last_trans = cur_trans
                        last_ori = cur_ori


                        time2 = time.time()
                        
                        # convert to tcp
                        trans = target_eepose[1][:3, 3]
                        ori = R.from_matrix(target_eepose[1][:3, :3]).as_quat(scalar_first=True)
                        new_trans = adjust_translation_along_quaternion(trans, ori, -0.1034)
                        target_eepose[1][:3, 3] = new_trans

                        if status == "ok":
                            # logger.debug(f"infer time:{time2 - time1}, trans: {new_trans}, ori: {cur_ori}")
                            real_robot.step(target_eepose, type="ee")
                        if status == "reset":
                            real_robot.reset()
                            status = "ok" if not want_to_pause else "pause"
                            break
                        elif status == "pause":
                            time.sleep(0.1)
                            continue
                        elif status == "end":
                            break
                        t += 1
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error: {e}")
                    # real_robot.reset()
                    status = "ok" if not want_to_pause else "pause"
                if status == "end":
                    break
        finally:
            print("end")




if __name__ == "__main__":
    # model
    model_path = "/data/share/yujunqiu/0628/checkpoints/steps_20000_pytorch_model.pt"
    unnorm_key = "place_0627_lmdb"
    # model_controller = ModelController(model_path, unnorm_key)
    model_controller = ModelController(debug=True, debug_path="./debug/pred_del_ee_grippers.npy", model_path=model_path, unnorm_key=unnorm_key)
    main(model_controller)