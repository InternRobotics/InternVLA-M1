import argparse
import h5py
import numpy as np
import time

from pathlib import Path
import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.getcwd())

import random
import cv2
import re
from scipy.spatial.transform import Rotation as R

from embodieddeploy.controller.franka_pandapy_impedance import FrankaController
from multiprocessing import Process, Lock, Pipe, Event, Queue
import pyspacemouse
from embodieddeploy.sensor.camera.realsense import MultiRealSenseCamera
from pdb import set_trace

def read_spacemouse(queue, lock):
    while True:
        state = pyspacemouse.read()
        with lock:
            if queue.full():
                queue.get()
            queue.put(state)

def clip_delta_min(delta, threshold):
    delta = np.where(abs(delta) < threshold, 0, delta)
    return delta

def get_hdf5_log_path(log_folder: Path):
    log_folder = Path(log_folder)
    files = os.listdir(log_folder)
    pattern = r'log-(\d{6})-\d{4}'
    existing_numbers = [int(re.match(pattern, file).group(1)) for file in files if re.match(pattern, file)]
    if not existing_numbers:
        next_number = 1
    else:
        existing_numbers.sort()
        next_number = existing_numbers[-1] + 1
    random_id = random.randint(1000, 9999)
    dir_path = log_folder / f"log-{next_number:06d}-{random_id}"
    os.makedirs(dir_path, exist_ok=True)
    new_filename = f"traj.hdf5"
    return dir_path / new_filename

def resize_image(x, w=224, h=224):
    # tmp_x = Image.fromarray(x)
    # tmp_x = tmp_x.resize((w, h))
    # return cv2.cvtColor(np.array(tmp_x), cv2.COLOR_RGB2BGR)
    return cv2.resize(x, (w,h), interpolation=cv2.INTER_LINEAR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data with Franka robot using SpaceMouse.")
    # 简写为-d或--dataset_path
    # parser.add_argument("-d", "--dataset_path", type=str, default="/home/pjlab/Desktop/data/0617_kitchen", help="Path to save the dataset.")
    parser.add_argument("-d", "--dataset_path", type=str, default="/data/yujunqiu/collected_data/place_0702/", help="Path to save the dataset.")
    args = parser.parse_args()
    root_dir = args.dataset_path
    print("Dataset will be saved in:", root_dir)

    use_camera = True
    ## define ##################
    collection_frq = 5
    max_queue_length = 5000
    translation_scale = 0.015
    rotation_scale = 3
    # control_mode = "pose"
    ############################

    ## initial --1##############
    # root_dir = "/home/pjlab/Desktop/data/0617_kitchen"
    log_folder = Path(root_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_folder, exist_ok=True)
    real_robot = FrankaController()
    if use_camera:
        camera = MultiRealSenseCamera()
    time.sleep(1)
    success = pyspacemouse.open()
    queue = Queue(maxsize=1)
    lock = Lock()
    process = Process(target=read_spacemouse, args=(queue, lock))
    process.start()
    ###########################

    ## initial data_saver #####
    # observation
    current_joints = []
    current_pose = []
    current_pose_quat = []
    current_gripper_width = []
    rgb0 = []
    rgb1 = []
    # depth0 = []
    # action
    actions_pose = []
    instruct_keypoint = []
    actions_pose_quat = []
    action_gripper_width = []
    # other information
    time_since_skill_started = []
    start_time = time.time()
    gripper_open = True
    step_counter = 0

    #############################
    # set a initial position
    # print("ee_pose_ini !!!")
    ee_pose_ini = np.array([[ 9.99999989e-01, -1.06083143e-05,  1.48668327e-04, 5.00965417e-01], [ 1.05955681e-05,  9.99999996e-01,  8.57367162e-05, 6.68212709e-03], [-1.48669236e-04, -8.57351400e-05,  9.99999985e-01, 2.93713933e-01], [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
    obs = real_robot.step((0.08, ee_pose_ini), type="pose", interplote=0,)
    time.sleep(3)
    print("ee_pose_ini finish !!!")
    ## store initial state
    # save state
    obs = real_robot.get_obs()
    ee_pose = obs["tcp_pose"]
    # current_joints.append(obs["current_joint"])
    # current_pose.append(obs["current_pose"])
    # current_pose_quat.append(obs["current_pose_quat"])
    # current_gripper_width.append(0.08)
    # time_since_skill_started.append(time.time() - start_time)

    if use_camera:
        # save images
        rgbd = camera.undistorted_rgbd()
        # rgb0.append(resize_image(rgbd[0][0]))
        rgb0.append(rgbd[0][0])
        rgb1.append(rgbd[0][1])
        # depth0.append(resize_image(rgbd[1][0]))
        cv2.imshow("Control Window", cv2.cvtColor(rgbd[0][1], cv2.COLOR_RGB2BGR))
    else:
        #生成一个空的图像窗口
        import numpy as np
        import cv2
        # cv2.namedWindow("Control Window", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Control Window", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow("Control Window", 640, 480)
        # cv2.imshow("Control Window", cv2.cvtColor(np.zeros((100,100,3)), cv2.COLOR_RGB2BGR))

    ##############################

    from collections import deque
    gripper_state_list = deque(maxlen=10)

    while True:
        trial_start = time.time()
        if not queue.empty():
            with lock:
                state = queue.get()
        else:
            continue
        op = cv2.waitKey(1) & 0xFF 
        
        gripper_state_list.append(gripper_open)


        translation_last = ee_pose[:3, 3]
        rotation = R.from_matrix(ee_pose[:3, :3])
        delta_translation = np.array([state.x * translation_scale, state.y * translation_scale, state.z * translation_scale])
        delta_translation = clip_delta_min(delta_translation, 0.1 * translation_scale)
        translation = translation_last + delta_translation
        if state.buttons[1]:
            rotation_delta = R.from_euler('yxz', [state.roll * rotation_scale, -state.pitch * rotation_scale, -state.yaw * rotation_scale], degrees=True)
            rotation = rotation_delta * rotation
        if state.buttons[0] or op == ord("g"):
            if len(set(gripper_state_list)) == 1:
                gripper_open = not gripper_open
        

        command = 0.08 if gripper_open else 0.0
        ee_pose = np.eye(4)
        ee_pose[:3, 3] = translation
        ee_pose[:3, :3] = rotation.as_matrix()
        ee_pose_quat = np.concatenate([translation, rotation.as_quat(scalar_first=True)]).tolist()
        # if False:
        #     action = (command, ee_pose_quat)
        #     obs = real_robot.step(action, type="pose_quat", interplote=0, read_gripper=state.buttons[0])
        action = (command, ee_pose)
        # save action
        actions_pose.append(ee_pose)
        instruct_keypoint.append(0)
        actions_pose_quat.append(ee_pose_quat)
        action_gripper_width.append(command)
        # apply action and get obs
        obs = real_robot.step(action, type="pose", interplote=0, read_gripper=state.buttons[0])

        # save state
        # current_joints.append(obs["current_joint"])
        # current_pose.append(obs["current_pose"])
        # current_pose_quat.append(obs["current_pose_quat"])
        # current_gripper_width.append(command)
        # time_since_skill_started.append(time.time() - start_time)
        # if use_camera:
        #     # save images
        #     rgbd = camera.undistorted_rgbd()
        #     # rgb0.append(resize_image(rgbd[0][0]))
        #     rgb0.append(rgbd[0][0])
        #     rgb1.append(rgbd[0][1])
        #     # depth0.append(resize_image(rgbd[1][0]))
        #     # from pdb import set_trace
        #     # set_trace()
        #     cv2.imshow("Control Window", cv2.cvtColor(rgbd[0][1], cv2.COLOR_RGB2BGR))
        
        # if len(rgb0) > max_queue_length:
        #     op = ord('r')        
        # print(f"collection time for one step: {time.time() - trial_start}")
        # sleep_time = max(0, 1/collection_frq - (time.time() - trial_start))
        # time.sleep(sleep_time)
        # step_time = time.time() - trial_start
        # print(f"after sleep, total time for one step: {step_time}, fps: {1 / step_time} \n")
        # step_counter += 1
        
        # if op == ord('q'):
        #     break

        # if op == ord('f'):
        #     instruct_keypoint[-1] = 1
        #     time.sleep(0.5)
        #     print()
        #     print("get a stop key")
        #     print()
        #     print("current instruct_keypoint:")
        #     print(instruct_keypoint)

        # if op == ord('s') or (state.buttons[0] and state.buttons[1]):
        #     # set_trace()
        #     print("total steps of this episode:", step_counter)
        #     h5_file = get_hdf5_log_path(log_folder)
        #     os.makedirs(os.path.dirname(h5_file), exist_ok=True)
        #     with h5py.File(h5_file, 'w') as f:
        #         f["actions_pose"] = np.array(actions_pose).astype(np.float32)
        #         f["actions_pose_quat"] = np.array(actions_pose_quat).astype(np.float32)
        #         f["instruct_keypoint"] = np.array(instruct_keypoint).astype(np.float32)
        #         f["action_gripper_width"] = np.array(action_gripper_width).astype(np.float32)
        #         f["current_joints"] = np.array(current_joints).astype(np.float32)[:-1]
        #         f["current_pose"] = np.array(current_pose).astype(np.float32)[:-1]
        #         f["current_pose_quat"] = np.array(current_pose_quat).astype(np.float32)[:-1]
        #         f["current_gripper_width"] = np.array(current_gripper_width).astype(np.float32)[:-1]
        #         f["time_since_skill_started"] = np.array(time_since_skill_started).astype(np.float32)[:-1]
        #         if use_camera:
        #             f["camera_1_image"] = np.array(rgb0)[:-1]
        #             f["camera_3_image"] = np.array(rgb1)[:-1]
        #         # f["camera_1_depth"] = np.expand_dims(np.array(depth0).astype(np.float32), axis=-1)[:-1]
        #     print("Save in {} Successfully.".format(h5_file))
        #     # reset robot
        #     real_robot.reset()
        #     time.sleep(2)
        #     print('You can collect now !!!')
        #     ## initial data_saver #####
        #     # observation
        #     current_joints = []
        #     current_pose = []
        #     current_pose_quat = []
        #     current_gripper_width = []
        #     rgb0 = []
        #     rgb1 = []
        #     # depth0 = []
        #     # action
        #     actions_pose = []
        #     actions_pose_quat = []
        #     instruct_keypoint = []
        #     action_gripper_width = []
        #     # other information
        #     time_since_skill_started = []
        #     start_time = time.time()
        #     gripper_open = True
        #     step_counter = 0
        #     print('wait.......')
        #     while True:
        #         op = cv2.waitKey(1) & 0xFF 
        #         if op == ord('h'):
        #             break
        #     print("ee_pose_ini !!!")
        #     ee_pose_ini = np.array([[ 9.99999989e-01, -1.06083143e-05,  1.48668327e-04, 5.00965417e-01], [ 1.05955681e-05,  9.99999996e-01,  8.57367162e-05, 6.68212709e-03], [-1.48669236e-04, -8.57351400e-05,  9.99999985e-01, 2.93713933e-01], [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
        #     obs = real_robot.step((0.08, ee_pose_ini), type="pose", interplote=0,)
        #     time.sleep(3)
        #     print("ee_pose_ini finish !!!")
        #     ## store initial state
        #     # save state
        #     obs = real_robot.get_obs()
        #     ee_pose = obs["current_pose"]
        #     current_joints.append(obs["current_joint"])
        #     current_pose.append(obs["current_pose"])
        #     current_pose_quat.append(obs["current_pose_quat"])
        #     current_gripper_width.append(0.08)
        #     time_since_skill_started.append(time.time() - start_time)
        #     if use_camera:
        #         # save images
        #         rgbd = camera.undistorted_rgbd()
        #         # rgb0.append(resize_image(rgbd[0][0]))
        #         rgb0.append(rgbd[0][0])
        #         rgb1.append(rgbd[0][1])
        #         # depth0.append(resize_image(rgbd[1][0]))
        #         cv2.imshow("Control Window", cv2.cvtColor(rgbd[0][1], cv2.COLOR_RGB2BGR))
        #         #############################
    
    real_robot.end()
    process.terminate()
    cv2.destroyAllWindows()
    exit(0)
