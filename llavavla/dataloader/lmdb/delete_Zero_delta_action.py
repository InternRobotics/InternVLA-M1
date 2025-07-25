import numpy as np
import lmdb
import pickle
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import os
import shutil

EPSILON = 5e-4

def delete_Zero_delta_arm_action(dataset_path):
    """
    删除零delta action的数据点，直接修改原始LMDB和PKL文件
    
    Args:
        dataset_path: 数据集路径
        episode_id: episode ID
    """
    lmdb_path = f"{dataset_path}/lmdb"
    meta_path = f"{dataset_path}/meta_info.pkl"
    
    # 创建备份
    backup_lmdb_path = f"{dataset_path}/lmdb_backup"
    backup_meta_path = f"{dataset_path}/meta_info_backup.pkl"
    
    if os.path.exists(backup_lmdb_path) and os.path.exists(backup_meta_path):
        print(f"Backup already exists at {backup_lmdb_path} and {backup_meta_path}")
        shutil.rmtree(backup_lmdb_path)
        os.remove(backup_meta_path)
        print(f"Backup deleted {backup_lmdb_path} and {backup_meta_path}")
        return
    
    print("Creating backup...")
    if os.path.exists(backup_lmdb_path):
        shutil.rmtree(backup_lmdb_path)
    shutil.copytree(lmdb_path, backup_lmdb_path)
    shutil.copy2(meta_path, backup_meta_path)
    
    # 打开原始LMDB环境
    lmdb_env = lmdb.open(
        lmdb_path, 
        readonly=True, 
        lock=False, 
        readahead=True, 
        meminit=True
    )
    
    # 加载meta信息
    meta_info = pickle.load(open(meta_path, "rb"))
    
    # 获取需要的数据键
    delta_ee_index = meta_info["keys"]["scalar_data"].index(b'delta_ee_pose_action')
    delta_ee_key = meta_info["keys"]["scalar_data"][delta_ee_index]
    
    gripper_index = meta_info["keys"]["scalar_data"].index(b'gripper_close')
    gripper_key = meta_info["keys"]["scalar_data"][gripper_index]
    
    # 获取所有需要保留的数据键
    scalar_keys = meta_info["keys"]["scalar_data"]
    primary_image_keys = meta_info["keys"][f"observation/obs_camera/color_image"]
    primary_image_2_keys = meta_info["keys"][f"observation/obs_camera_2/color_image"]
    wrist_image_keys = meta_info["keys"]["observation/realsense/color_image"]
    
    # 读取delta_ee_pose_action和gripper数据
    with lmdb_env.begin(write=False) as txn:
        delta_ee_actions = pickle.loads(txn.get(delta_ee_key))
        gripper_actions = pickle.loads(txn.get(gripper_key))
    
    # 计算非零delta indices
    non_zero_indices = []
    previous_gripper = 1.0  # 初始gripper状态
    
    for i in range(len(delta_ee_actions)):
        delta_ee = delta_ee_actions[i]
        gripper = gripper_actions[i]
        
        # 检查delta_ee_pose_action是否为零
        # delta_ee是[translation(3), quaternion(4)]的形式
        translation_delta = delta_ee[0]  # 第一个array是translation (3,)
        quaternion_delta = delta_ee[1]   # 第二个array是quaternion (4,)
        
        # 将四元数转换为欧拉角来检查旋转变化
        rotation_delta = R.from_quat(quaternion_delta, scalar_first=True).as_euler('xyz', degrees=False)
        
        # 检查平移和旋转是否有显著变化
        has_translation_change = np.any(np.abs(translation_delta) > EPSILON)
        has_rotation_change = np.any(np.abs(rotation_delta) > EPSILON)
        
        # 检查gripper状态是否改变
        has_gripper_change = gripper != (previous_gripper if i == 0 else gripper_actions[i-1])
        
        # 如果有任何变化，保留这个数据点
        if has_translation_change or has_gripper_change:
            non_zero_indices.append(i)
        else:
            # print(f"Removing data point {i} because it has no significant changes")
            pass
        
        previous_gripper = gripper
    
    print(f"Original length: {len(delta_ee_actions)}")
    print(f"Filtered length: {len(non_zero_indices)}")
    print(f"Removed {len(delta_ee_actions) - len(non_zero_indices)} zero-delta actions")
    
    # 关闭原始环境
    lmdb_env.close()
    
    # 删除原始LMDB目录
    shutil.rmtree(lmdb_path)
    
    # 创建新的LMDB环境（使用原始路径）
    new_lmdb_env = lmdb.open(
        lmdb_path,
        map_size=1024**4,  # 1TB
        readonly=False,
        meminit=False,
        map_async=True,
    )
    
    # 重新打开备份环境读取数据
    backup_lmdb_env = lmdb.open(
        backup_lmdb_path,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=True
    )
    
    # 复制过滤后的数据到新环境
    with backup_lmdb_env.begin(write=False) as read_txn, new_lmdb_env.begin(write=True) as write_txn:
        # 处理scalar数据
        for key in scalar_keys:
            original_data = pickle.loads(read_txn.get(key))
            filtered_data = [original_data[i] for i in non_zero_indices]
            write_txn.put(key, pickle.dumps(filtered_data))
        
        # 处理图像数据 - Primary camera images
        new_primary_keys = []
        for new_idx, old_idx in enumerate(non_zero_indices):
            old_key = primary_image_keys[old_idx]
            # 创建新的键，使用连续的索引
            new_key = f"observation/obs_camera/color_image/{new_idx:06d}".encode()
            new_primary_keys.append(new_key)
            image_data = read_txn.get(old_key)
            write_txn.put(new_key, image_data)

        new_primary_2_keys = []
        for new_idx, old_idx in enumerate(non_zero_indices):
            old_key = primary_image_2_keys[old_idx]
            new_key = f"observation/obs_camera_2/color_image/{new_idx:06d}".encode()
            new_primary_2_keys.append(new_key)
            image_data = read_txn.get(old_key)
            write_txn.put(new_key, image_data)
        
        # 处理图像数据 - Wrist camera images
        new_wrist_keys = []
        for new_idx, old_idx in enumerate(non_zero_indices):
            old_key = wrist_image_keys[old_idx]
            # 创建新的键，使用连续的索引
            new_key = f"observation/realsense/color_image/{new_idx:06d}".encode()
            new_wrist_keys.append(new_key)
            image_data = read_txn.get(old_key)
            write_txn.put(new_key, image_data)
    
    # 关闭环境
    backup_lmdb_env.close()
    new_lmdb_env.close()
    
    # 更新meta信息
    meta_info["keys"][f"observation/obs_camera/color_image"] = new_primary_keys
    meta_info["keys"][f"observation/obs_camera_2/color_image"] = new_primary_2_keys
    meta_info["keys"]["observation/realsense/color_image"] = new_wrist_keys
    meta_info["num_steps"] = len(non_zero_indices)
    
    # 保存更新后的meta信息到原始路径
    with open(meta_path, "wb") as f:
        pickle.dump(meta_info, f)
    
    # 删除备份文件
    shutil.rmtree(backup_lmdb_path)
    os.remove(backup_meta_path)
    print(f"Backup deleted {backup_lmdb_path} and {backup_meta_path}")

    # print(f"Original LMDB updated at {lmdb_path}")
    # print(f"Original meta info updated at {meta_path}")
    # print(f"Backup saved at {backup_lmdb_path} and {backup_meta_path}")

# 使用示例
if __name__ == "__main__":
    dataset_path = "/shared/smartbot/genmanip/demonstrations/testDelte/2025-06-15_14_30_29_658059"
    delete_Zero_delta_arm_action(dataset_path)