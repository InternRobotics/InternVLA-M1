import os
import pickle
import json 
import numpy as np
import lmdb 
import glob 
from scipy.spatial.transform import Rotation
from pdb import set_trace 
from tqdm import tqdm 
from multiprocessing import Pool
import argparse
from pathlib import Path
from delete_Zero_delta_action_realdata import delete_Zero_delta_arm_action

def parse_args():
    parser = argparse.ArgumentParser(description='Process real dataset with various options')
    
    parser.add_argument('-d', '--dataset_names', type=str, nargs='+', required=True,
                      help='List of dataset names separated by space')
    
    # Create mutually exclusive group for sequence handling
    sequence_group = parser.add_mutually_exclusive_group()
    sequence_group.add_argument('-s', '--sequence_nums', type=int, default=None,
                            help='Number of sequences to process')
    sequence_group.add_argument('-a', '--all_sequence', action='store_true',
                            help='Process all sequences')
    
    parser.add_argument('-m', '--multiprocess', action='store_true',
                      help='Use multiprocessing version of functions')
    parser.add_argument('-n', '--num_processes', type=int, default=48,
                      help='Number of processes to use')
    
    parser.add_argument('-g', '--is_debug', action='store_true',
                      help='Is debug')
    parser.add_argument('-z', '--delete_zero_delta_arm_action', action='store_true',
                      help='Delete zero delta arm action')

    return parser.parse_args()

def make_data_info(root_path, save_json_path, save_meta_path, sequence_num=None):
    data_list = glob.glob(os.path.join(root_path, f"*"))
    data_list.sort()
    if sequence_num is not None:
        data_list = data_list[:sequence_num]
    save_data_list = []
    qpos_list = []
    gripper_close_list = []
    abs_eepose_action_list = []
    delta_eepose_action_list = []
    ee_pose_state_list = []
    cnt = 0 
    for data_path in tqdm(data_list):
        lmdb_env = lmdb.open(
            f"{data_path}/lmdb", 
            readonly=True, 
            lock=False, 
            readahead=False, 
            meminit=False
        )
        meta_info = pickle.load(
            open(
            f"{data_path}/meta_info.pkl", 
            "rb"
            )
        )
        
        # Get keys for real data format
        qpos_key = b'observation/robot/qpos'
        gripper_close_key = b'gripper_close'
        abs_ee_pose_action_key = b'ee_pose_action'
        delta_ee_pose_action_key = b'delta_ee_pose_action'
        ee_pose_state_key = b'observation/robot/ee_pose_state'

        try:
            with lmdb_env.begin(write=False) as txn:
                qpos = pickle.loads(txn.get(qpos_key))
                gripper_close = pickle.loads(txn.get(gripper_close_key))
                abs_ee_pose_action = pickle.loads(txn.get(abs_ee_pose_action_key))
                delta_ee_pose_action = pickle.loads(txn.get(delta_ee_pose_action_key))
                ee_pose_state = pickle.loads(txn.get(ee_pose_state_key))

        except Exception as e:
            print(f"Error loading data for {data_path}: {e}")
            continue

        qpos_list += qpos.tolist()
        gripper_close_list += gripper_close.tolist()
        abs_eepose_action_list += abs_ee_pose_action.tolist()
        delta_eepose_action_list += delta_ee_pose_action.tolist()
        ee_pose_state_list += ee_pose_state.tolist()
        
        num_steps = len(qpos)
        episode_id = data_path.split("/")[-1]
        save_data = [episode_id, num_steps]
        save_data_list.append(save_data)
    
    with open(save_json_path, 'w') as file:
        json.dump(save_data_list, file, indent=4)

    # Stack and compute statistics for qpos
    qpos_list = np.stack(qpos_list)
    qpos_mean = np.mean(qpos_list, axis=0)
    qpos_std = np.std(qpos_list, axis=0) 
    qpos_min = np.min(qpos_list, axis=0)
    qpos_max = np.max(qpos_list, axis=0)
    qpos_q01 = np.percentile(qpos_list, 1, axis=0)
    qpos_q99 = np.percentile(qpos_list, 99, axis=0)

    # Stack and compute statistics for gripper close
    gripper_close_list = np.stack(gripper_close_list)
    gripper_close_mean = np.mean(gripper_close_list, axis=0)
    gripper_close_std = np.std(gripper_close_list, axis=0)
    gripper_close_min = np.min(gripper_close_list, axis=0)
    gripper_close_max = np.max(gripper_close_list, axis=0)
    gripper_close_q01 = np.percentile(gripper_close_list, 1, axis=0)
    gripper_close_q99 = np.percentile(gripper_close_list, 99, axis=0)

    # Create metadata dictionary with all statistics
    meta_info = {
        "abs_qpos_mean": qpos_mean.tolist(),
        "abs_qpos_std": qpos_std.tolist(),
        "abs_qpos_min": qpos_min.tolist(),
        "abs_qpos_max": qpos_max.tolist(),
        "abs_qpos_q01": qpos_q01.tolist(),
        "abs_qpos_q99": qpos_q99.tolist(),
        "gripper_close_mean": gripper_close_mean.tolist(),
        "gripper_close_std": gripper_close_std.tolist(),
        "gripper_close_min": gripper_close_min.tolist(),
        "gripper_close_max": gripper_close_max.tolist(),
        "gripper_close_q01": gripper_close_q01.tolist(),
        "gripper_close_q99": gripper_close_q99.tolist(),
    }
    
    # Add ee pose action statistics
    meta_info["abs_eepose_action_mean"] = np.mean(abs_eepose_action_list, axis=0).tolist()
    meta_info["abs_eepose_action_std"] = np.std(abs_eepose_action_list, axis=0).tolist()
    meta_info["abs_eepose_action_min"] = np.min(abs_eepose_action_list, axis=0).tolist()
    meta_info["abs_eepose_action_max"] = np.max(abs_eepose_action_list, axis=0).tolist()
    meta_info["abs_eepose_action_q01"] = np.percentile(abs_eepose_action_list, 1, axis=0).tolist()
    meta_info["abs_eepose_action_q99"] = np.percentile(abs_eepose_action_list, 99, axis=0).tolist()
    meta_info["delta_eepose_action_mean"] = np.mean(delta_eepose_action_list, axis=0).tolist() 
    meta_info["delta_eepose_action_std"] = np.std(delta_eepose_action_list, axis=0).tolist()
    meta_info["delta_eepose_action_min"] = np.min(delta_eepose_action_list, axis=0).tolist()
    meta_info["delta_eepose_action_max"] = np.max(delta_eepose_action_list, axis=0).tolist()
    meta_info["delta_eepose_action_q01"] = np.percentile(delta_eepose_action_list, 1, axis=0).tolist()
    meta_info["delta_eepose_action_q99"] = np.percentile(delta_eepose_action_list, 99, axis=0).tolist()
    meta_info["eepose_state_mean"] = np.mean(ee_pose_state_list, axis=0).tolist()
    meta_info["eepose_state_std"] = np.std(ee_pose_state_list, axis=0).tolist()
    meta_info["eepose_state_min"] = np.min(ee_pose_state_list, axis=0).tolist()
    meta_info["eepose_state_max"] = np.max(ee_pose_state_list, axis=0).tolist()
    meta_info["eepose_state_q01"] = np.percentile(ee_pose_state_list, 1, axis=0).tolist()
    meta_info["eepose_state_q99"] = np.percentile(ee_pose_state_list, 99, axis=0).tolist()

    # Save metadata to pickle file
    pickle.dump(meta_info, open(save_meta_path, "wb"))

    # Print key statistics
    print("abs_qpos_mean:", qpos_mean)
    print("abs_qpos_std:", qpos_std)
    print("abs_qpos_q01:", qpos_q01)
    print("abs_qpos_q99:", qpos_q99)
    print("delta_eepose_action_mean:", meta_info["delta_eepose_action_mean"])
    print("delta_eepose_action_std:", meta_info["delta_eepose_action_std"])
    print("delta_eepose_action_min:", meta_info["delta_eepose_action_min"])
    print("delta_eepose_action_max:", meta_info["delta_eepose_action_max"])
    print("delta_eepose_action_q01:", meta_info["delta_eepose_action_q01"])
    print("delta_eepose_action_q99:", meta_info["delta_eepose_action_q99"])
    print("save_meta_path:", save_meta_path)

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def process_data_path(data_path):
    """Worker function to process a single data path for multiprocessing."""
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", 
        readonly=True, 
        lock=False, 
        readahead=False, 
        meminit=False
    )
    try:
        meta_info = pickle.load(
            open(
            f"{data_path}/meta_info.pkl", 
            "rb"
            )
        )
    except Exception as e:
        print(f"Error loading meta_info for {data_path}: {e}")
        return None
    
    # Get keys for real data format
    qpos_key = b'observation/robot/qpos'
    gripper_close_key = b'gripper_close'
    abs_ee_pose_action_key = b'ee_pose_action'
    delta_ee_pose_action_key = b'delta_ee_pose_action'
    ee_pose_state_key = b'observation/robot/ee_pose_state'
    
    try:
        with lmdb_env.begin(write=False) as txn:
            qpos = pickle.loads(txn.get(qpos_key))
            gripper_close = pickle.loads(txn.get(gripper_close_key))
            abs_ee_pose_action = pickle.loads(txn.get(abs_ee_pose_action_key))
            delta_ee_pose_action = pickle.loads(txn.get(delta_ee_pose_action_key))
            ee_pose_state = pickle.loads(txn.get(ee_pose_state_key))
    except Exception as e:
        print(f"Error loading data for {data_path}: {e}")
        return None
    
    num_steps = len(qpos)
    episode_id = data_path.split("/")[-1]
    save_data = [episode_id, num_steps]
    
    return {
        'save_data': save_data,
        'qpos': qpos.tolist(),
        'gripper_close': gripper_close.tolist(),
        'abs_eepose_action': abs_ee_pose_action.tolist(),
        'delta_eepose_action': delta_ee_pose_action.tolist(),
        'ee_pose_state': ee_pose_state.tolist()
    }

def process_data_path_with_timeout(args):
    data_path, timeout = args
    
    # 使用超时机制调用原始函数
    import signal
    
    def handler(signum, frame):
        raise TimeoutError(f"Processing {data_path} timed out")
    
    # 设置信号处理器和闹钟
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    
    try:
        result = process_data_path(data_path)
        signal.alarm(0)  # 关闭闹钟
        return result
    except TimeoutError as e:
        print(str(e))
        return None

def make_data_info_multiprocess(root_path, save_json_path, save_meta_path, sequence_num=None, num_processes=48):
    """Multiprocessing version of make_data_info function for real data."""
    data_list = glob.glob(os.path.join(root_path, f"*"))
    data_list.sort()
    if sequence_num is not None:
        data_list = data_list[:sequence_num]
    
    # Initialize empty lists
    save_data_list = []
    qpos_list = []
    gripper_close_list = []
    abs_eepose_action_list = []
    delta_eepose_action_list = []
    ee_pose_state_list = []
    
    # Create a pool of workers
    with Pool(processes=num_processes) as pool:
        # Map the worker function to all data paths
        results = list(tqdm(pool.imap(process_data_path_with_timeout, [(data_path, 10) for data_path in data_list]), total=len(data_list)))
    
    # Collect results from all processes
    for result in results:
        # Add null check to prevent errors with None results
        if result is None:
            print("Warning: Received None result from a worker process")
            continue
        
        try:
            save_data_list.append(result['save_data'])
            qpos_list.extend(result['qpos'])
            gripper_close_list.extend(result['gripper_close'])
            abs_eepose_action_list.extend(result['abs_eepose_action'])
            delta_eepose_action_list.extend(result['delta_eepose_action'])
            ee_pose_state_list.extend(result['ee_pose_state'])
        except Exception as e:
            print(f"Error processing result: {e}")
            continue
    
    # Check if we have any data to process
    if not qpos_list:
        print("Error: No valid data was collected from any worker process")
        return
    
    # Save data to JSON file
    with open(save_json_path, 'w') as file:
        json.dump(save_data_list, file, indent=4)

    # Stack and compute statistics for qpos
    qpos_list = np.stack(qpos_list)
    qpos_mean = np.mean(qpos_list, axis=0)
    qpos_std = np.std(qpos_list, axis=0) 
    qpos_min = np.min(qpos_list, axis=0)
    qpos_max = np.max(qpos_list, axis=0)
    qpos_q01 = np.percentile(qpos_list, 1, axis=0)
    qpos_q99 = np.percentile(qpos_list, 99, axis=0)

    # Stack and compute statistics for gripper close
    gripper_close_list = np.stack(gripper_close_list)
    gripper_close_mean = np.mean(gripper_close_list, axis=0)
    gripper_close_std = np.std(gripper_close_list, axis=0)
    gripper_close_min = np.min(gripper_close_list, axis=0)
    gripper_close_max = np.max(gripper_close_list, axis=0)
    gripper_close_q01 = np.percentile(gripper_close_list, 1, axis=0)
    gripper_close_q99 = np.percentile(gripper_close_list, 99, axis=0)

    # Create metadata dictionary with all statistics
    meta_info = {
        "abs_qpos_mean": qpos_mean.tolist(),
        "abs_qpos_std": qpos_std.tolist(),
        "abs_qpos_min": qpos_min.tolist(),
        "abs_qpos_max": qpos_max.tolist(),
        "abs_qpos_q01": qpos_q01.tolist(),
        "abs_qpos_q99": qpos_q99.tolist(),
        "gripper_close_mean": gripper_close_mean.tolist(),
        "gripper_close_std": gripper_close_std.tolist(),
        "gripper_close_min": gripper_close_min.tolist(),
        "gripper_close_max": gripper_close_max.tolist(),
        "gripper_close_q01": gripper_close_q01.tolist(),
        "gripper_close_q99": gripper_close_q99.tolist(),
    }
    
    # Add ee pose action statistics
    meta_info["abs_eepose_action_mean"] = np.mean(abs_eepose_action_list, axis=0).tolist()
    meta_info["abs_eepose_action_std"] = np.std(abs_eepose_action_list, axis=0).tolist()
    meta_info["abs_eepose_action_min"] = np.min(abs_eepose_action_list, axis=0).tolist()
    meta_info["abs_eepose_action_max"] = np.max(abs_eepose_action_list, axis=0).tolist()
    meta_info["abs_eepose_action_q01"] = np.percentile(abs_eepose_action_list, 1, axis=0).tolist()
    meta_info["abs_eepose_action_q99"] = np.percentile(abs_eepose_action_list, 99, axis=0).tolist()
    meta_info["delta_eepose_action_mean"] = np.mean(delta_eepose_action_list, axis=0).tolist() 
    meta_info["delta_eepose_action_std"] = np.std(delta_eepose_action_list, axis=0).tolist()
    meta_info["delta_eepose_action_min"] = np.min(delta_eepose_action_list, axis=0).tolist()
    meta_info["delta_eepose_action_max"] = np.max(delta_eepose_action_list, axis=0).tolist()
    meta_info["delta_eepose_action_q01"] = np.percentile(delta_eepose_action_list, 1, axis=0).tolist()
    meta_info["delta_eepose_action_q99"] = np.percentile(delta_eepose_action_list, 99, axis=0).tolist()
    meta_info["eepose_state_mean"] = np.mean(ee_pose_state_list, axis=0).tolist()
    meta_info["eepose_state_std"] = np.std(ee_pose_state_list, axis=0).tolist()
    meta_info["eepose_state_min"] = np.min(ee_pose_state_list, axis=0).tolist()
    meta_info["eepose_state_max"] = np.max(ee_pose_state_list, axis=0).tolist()
    meta_info["eepose_state_q01"] = np.percentile(ee_pose_state_list, 1, axis=0).tolist()
    meta_info["eepose_state_q99"] = np.percentile(ee_pose_state_list, 99, axis=0).tolist()

    # Save metadata to pickle file
    pickle.dump(meta_info, open(save_meta_path, "wb"))

    # Print key statistics
    print("abs_qpos_mean:", qpos_mean)
    print("abs_qpos_std:", qpos_std)
    print("abs_qpos_q01:", qpos_q01)
    print("abs_qpos_q99:", qpos_q99)
    print("delta_eepose_action_mean:", meta_info["delta_eepose_action_mean"])
    print("delta_eepose_action_std:", meta_info["delta_eepose_action_std"])
    print("delta_eepose_action_min:", meta_info["delta_eepose_action_min"])
    print("delta_eepose_action_max:", meta_info["delta_eepose_action_max"])
    print("delta_eepose_action_q01:", meta_info["delta_eepose_action_q01"])
    print("delta_eepose_action_q99:", meta_info["delta_eepose_action_q99"])
    print("save_meta_path:", save_meta_path)

def delete_zero_delta_arm_action(dataset_path, multiprocess=False):
    """删除所有delta_ee_pose_action为零的数据"""
    data_list = glob.glob(os.path.join(dataset_path, f"*"))
    # data_list.sort()
    print("delete zero delta arm action")
    if multiprocess:
        with Pool(processes=48) as pool:
            list(tqdm(pool.imap(delete_Zero_delta_arm_action, data_list), total=len(data_list)))
    else:
        for data_path in tqdm(data_list):
            delete_Zero_delta_arm_action(data_path)


if __name__ == "__main__":
    args = parse_args()
    if args.is_debug:
        import debugpy 
        debugpy.listen(("0.0.0.0", 10092))  # 监听端口 
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()  # 等待 VS Code 附加
    root_path = "/mnt/petrelfs/share/efm_p/wangfangjing/datasets"
    make_dir(f"{root_path}/data_info/")
    
    for dataset_name in args.dataset_names:
        dataset_path = f"{root_path}/{dataset_name}/render"

        if args.delete_zero_delta_arm_action:
            delete_zero_delta_arm_action(dataset_path, multiprocess=args.multiprocess)

        if args.sequence_nums is not None:
            sequence_num = args.sequence_nums
            save_json_path = f"{root_path}/data_info/{dataset_name}_{sequence_num}.json"
            save_meta_path = f"{root_path}/data_info/{dataset_name}_{sequence_num}.pkl"
            Path(save_json_path).parent.mkdir(parents=True, exist_ok=True, mode=0o777)
        else:
            sequence_num = None
            save_json_path = f"{root_path}/data_info/{dataset_name}.json"
            save_meta_path = f"{root_path}/data_info/{dataset_name}.pkl"
            Path(save_json_path).parent.mkdir(parents=True, exist_ok=True, mode=0o777)

        # Use multiprocessing version if requested
        if args.multiprocess:
            make_data_info_multiprocess(dataset_path, save_json_path, save_meta_path, sequence_num, args.num_processes)
        else:
            make_data_info(dataset_path, save_json_path, save_meta_path, sequence_num)
