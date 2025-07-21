import argparse
import h5py
import numpy as np
import time

# from autolab_core import RigidTransform
from pathlib import Path
import os
import random
import cv2
import re
from scipy.spatial.transform import Rotation as R
from embodieddeploy.controller.panda_real_robot import PandaRealRobot
from multiprocessing import Process, Lock, Pipe, Event, Queue
import pyspacemouse
# from pynput import keyboard
# from pynput.keyboard import Key, Listener
from panda_py import libfranka
import asyncio


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

def create_formated_skill_dict(joints, end_effector_positions, time_since_skill_started):
    skill_dict = dict(skill_description='GuideMode', skill_state_dict=dict())
    skill_dict['skill_state_dict']['q'] = np.array(joints)
    skill_dict['skill_state_dict']['O_T_EE'] = np.array(end_effector_positions)
    skill_dict['skill_state_dict']['time_since_skill_started'] = np.array(time_since_skill_started)

    # The key (0 here) usually represents the absolute time when the skill was started but
    formatted_dict = {0: skill_dict}
    return formatted_dict

def get_log_folder(log_root: str):
    log_folder = Path(log_root) / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_folder, exist_ok=True)
    return log_folder

def get_json_log_path(log_folder: Path):
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
    new_filename = f"traj.json"
    return dir_path / new_filename

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

def save_color_images(colors, images_dir, now_time):
    for i in range(len(colors)):
        os.makedirs(images_dir / f"color_{i}", exist_ok=True)
        cv2.imwrite(str(images_dir / f"color_{i}" / f"{now_time}.png"), cv2.cvtColor(colors[i], cv2.COLOR_RGB2BGR))
        # cv2.imshow(f"Color {i}", cv2.cvtColor(colors[i], cv2.COLOR_RGB2BGR))

def save_depth_images(depths, images_dir, now_time):
    for i in range(len(depths)):
        os.makedirs(images_dir / f"depth_{i}", exist_ok=True)
        np.save(str(images_dir / f"depth_{i}" / f"{now_time}.npy"), depths[i])

def save_images(images, images_dir, headless=True):
    tmp_colors, tmp_depths = images
    now_time = int(time.time() * 1000)
    save_color_images(tmp_colors, images_dir, now_time)
    save_depth_images(tmp_depths, images_dir, now_time)

    if not headless:
        for i in range(len(tmp_colors)):
            # turn depth into colorful
            depth = tmp_depths[i]
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)
            # cv2.imshow(f"Color {i}", cv2.cvtColor(tmp_colors[i], cv2.COLOR_RGB2BGR))
            # cv2.imshow(f"Depth {i}", depth)
    return now_time

def save_dict_to_hd5f(group, dict_data):
    for key, value in dict_data.items():
        if isinstance(value, dict):
            save_dict_to_hd5f(group.create_group(key), value)
        else:
            group.create_dataset(key, data=value)



def save_images_process(image_queue, stop_event, save_event):
    """
    子进程函数，用于保存图像数据。
    """
    try:
        while not stop_event.is_set():
            if image_queue.qsize() > 0:  # 检查队列中是否有数据
                # with lock:
                if not save_event.is_set():
                    (images, images_dir) = image_queue.get(timeout=1)
                    now_time = int(time.time() * 1000)
                    save_color_images(images[0], images_dir, now_time)
                    save_depth_images(images[1], images_dir, now_time)

    except Exception as e:
        print(f"Error in save_images_process: {e}")


if __name__ == "__main__":
    save_flag = 0
    
    translation_scale = 0.005
    rotation_scale = 1
    gripper_open = True
    # data_dir = "/data/hanxiaoshen/real-data/place_on_the_board"
    data_dir = "/data/yujunqiu/collected_data/place_0702"
    data_dir = get_log_folder(data_dir)
    
    i = 0
    real_robot = PandaRealRobot()
    multi_camera = real_robot.cameras
    time.sleep(1)
    obs = real_robot.get_obs()
    ee_pose = obs["panda_hand_pose"]
    success = pyspacemouse.open()
    queue = Queue(maxsize=1)
    lock = Lock()
    process = Process(target=read_spacemouse, args=(queue, lock))
    process.start()
    
    # init saver
    end_effector_position = []
    joints = []
    actions = []
    gripper_width = []
    rgb0 = []
    rgb1 = []
    rgb2 = []
    # rgb3 = []
    depth0 = []
    depth1 = []
    depth2 = []
    # depth3 = []
    h5_file = get_hdf5_log_path(data_dir)
    time_since_skill_started = []
    timestames = []
    start_time = time.time()
    last_time = time.time()
    save_images_processes = []
    
    
    
    if save_flag:
        # image_queue = Queue(maxsize=10000)
        num_process = 4
        image_queue = Queue(maxsize=1000)
        stop_event = Event()
        save_lock = Lock()
        save_event = Event()
        save_processes = [
            Process(target=save_images_process, args=(image_queue, stop_event, save_event))
            for i in range(num_process)
        ]
        for p in save_processes:
            p.start()


    # try:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow("Control Window", img)
    
    while True:
        start = time.time()
        if not queue.empty():
            with lock:
                state = queue.get() # SpaceNavigator(t=-1, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, buttons=[0, 0])
        else:
            continue
        op = cv2.waitKey(1) & 0xFF 
        

        translation = ee_pose[:3, 3]
        rotation = R.from_matrix(ee_pose[:3, :3])
        delta_translation = np.array([state.x * translation_scale, state.y * translation_scale, state.z * translation_scale])
        delta_translation = clip_delta_min(delta_translation, 0.1 * translation_scale)
        translation = translation + delta_translation
        if state.buttons[1]:
            rotation_delta = R.from_euler('yxz', [state.roll * rotation_scale, -state.pitch * rotation_scale, -state.yaw * rotation_scale], degrees=True)
            rotation = rotation_delta * rotation
        if state.buttons[0]:
            gripper_open = not gripper_open
        ee_pose = np.eye(4)
        ee_pose[:3, 3] = translation
        ee_pose[:3, :3] = rotation.as_matrix()
        command = 0.08 if gripper_open else 0.0

        action = (command, ee_pose)
        obs = real_robot.step(action, type="ee", interplote=0, read_gripper=state.buttons[0])
        
        # collect data
        end_effector_position.append(ee_pose)
        robot_state = obs["state"]
        # print("state: ",robot_state)
        tmp_gripper_width = sum(robot_state[7:])
        # 将robot——state最后两个元素替换为command/2
        tmp_action = np.concatenate((robot_state[:7] ,[command/2, command/2]))
        actions.append(tmp_action)
        # print(tmp_action[-2:])
        joints.append(robot_state)
        time_since_skill_started.append(time.time() - start_time)
        timestames.append(time.time())
        
        # save images
        time05 = time.time()
        rgbd = real_robot.cameras.undistorted_rgbd()
        time07 = time.time()
        
        rgb0.append(rgbd[0][0])
        rgb1.append(rgbd[0][1])
        # rgb2.append(rgbd[0][2])
        # rgb3.append(rgbd[0][0])
        depth0.append(rgbd[1][0])
        depth1.append(rgbd[1][1])
        # depth2.append(rgbd[1][2])
        # depth3.append(rgbd[1][0])
        
        if save_flag:
            # try:
            file = h5_file.parent
            # rgbd = (encode_image_RGB(rgbd[0]), encode_image_depth(rgbd[1]))
            data = (rgbd, file)
            # with save_lock:
            save_event.set()
            image_queue.put(data)
            save_event.clear()
            # except Exception as e:
            #     print("save exception: ",e)
        time08 = time.time()
        
        # time_stamp = save_images(rgbd, h5_file.parent, headless=False)
        # print(f"time for img: {time07 - time05}")
        # print(f"time for img_queue: {time08 - time07}")
        i += 1
        time.sleep(max(0, 1/30 - (time.time() - start)))
        
        
        if op == ord('q'): # quit
            # process.terminate()
            if save_flag:
                stop_event.set()
            # process_save_images_2.terminate()
            # process_save_images.terminate()
            # process_save_images_2.join()
            # process_save_images.join()
            break
        elif op == ord('s') or (state.buttons[0] and state.buttons[1]):
            # stop_event.set()
            # process_save_images.join()
            
            with h5py.File(h5_file, 'w') as f:
                f["action"] = np.array(actions[:-1])
                print(actions[:-1])
                f.create_group("observations")
                # 转换为 <f4
                f["observations/qpos"] = np.array(joints).astype(np.float32)
                f["observations/ee_pose"] = np.array(end_effector_position)
                f["observations/time_since_skill_started"] = np.array(time_since_skill_started)
                f["observations/timestamp"] = np.array(timestames)
                # intr_colors = multi_camera.get_intrinsic_color()
                # intr_depths = multi_camera.get_intrinsic_depth()
                f.create_group("observations/images")
                for i in range(3):
                    if i == 2:
                        f[f"observations/images/wrist_camera"] = np.array(eval(f"rgb{i}"))
                        f[f"observations/depths/wrist_camera"] = np.array(eval(f"depth{i}")).astype(np.float32)
                    else:
                        f[f"observations/images/camera_{i}"] = np.array(eval(f"rgb{i}"))
                        f[f"observations/depths/camera_{i}"] = np.array(eval(f"depth{i}")).astype(np.float32)

            # reset robot
            real_robot.reset_robot() 
            obs = real_robot.get_obs()
            
            ee_pose = obs["panda_hand_pose"]
            
            end_effector_position = []
            joints = []
            actions = []
            gripper_width = []
            h5_file = get_hdf5_log_path(data_dir)
            multi_camera = real_robot.cameras
            time_since_skill_started = []
            timestames = []
            start_time = time.time()
            last_time = time.time()
            gripper_open = True
            # 清空rgb和depth
            rgb0 = []
            rgb1 = []
            rgb2 = []
            depth0 = []
            depth1 = []
            depth2 = []
            print(i, " steps")
            i = 0
            print("Save in {} Successfully.".format(h5_file))

            while True:
                op = cv2.waitKey(1) & 0xFF 
                if op == ord('c'):
                    break
        elif op == ord('c'):
            # reset robot
            real_robot.reset_robot() 
            obs = real_robot.get_obs()
            
            ee_pose = obs["panda_hand_pose"]
            
            end_effector_position = []
            joints = []
            actions = []
            gripper_width = []
            h5_file = get_hdf5_log_path(data_dir)
            multi_camera = real_robot.cameras
            time_since_skill_started = []
            timestames = []
            start_time = time.time()
            last_time = time.time()
            gripper_open = True
            # 清空rgb和depth
            rgb0 = []
            rgb1 = []
            rgb2 = []
            depth0 = []
            depth1 = []
            depth2 = []
            print(i, " steps")
            i = 0
            print("Save in {} Successfully.".format(h5_file))

            while True:
                op = cv2.waitKey(1) & 0xFF 
                if op == ord('c'):
                    break
    # except Exception as e:
    #     print(e)
    
    # finally:
    
    print(i, " steps")
    real_robot.end()
    # process_save_images.terminate()
    # process_save_images_2.terminate()
    if save_flag:
        for p in save_processes:
            p.join()
    process.terminate()
    # process.close()
    # process_save.close()
    cv2.destroyAllWindows()
    exit(0)
