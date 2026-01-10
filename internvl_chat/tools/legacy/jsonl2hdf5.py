import os
import json
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_rgb(path):
    return np.asarray(Image.open(path))

def load_depth(path):
    return np.load(path)

def main(jsonl_path, output_path):
    img_path_prefix = "/SSD_DISK/users/zhangjiahui/openvla/data/franka/replay_traj_sampled_task_mv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    with h5py.File(output_path, 'w') as h5file:
        data_group = h5file.create_group('data')

        for idx, line in tqdm(enumerate(lines), total=len(lines)):
            item = json.loads(line.strip())
            episode_id = str(idx)
            ep_group = data_group.create_group(episode_id)

            # actions
            ep_group.create_dataset("actions", data=np.array(item['action'], dtype=np.float32))


            obs_group = ep_group.create_group("obs")

            # RGB images
            # import pdb; pdb.set_trace()
            # mode = item['mode']
            # agentview_rgbs = np.stack([load_rgb(os.path.join(img_path_prefix, mode, p)) for p in item['image']], axis=0)
            agentview_rgbs = np.stack([load_rgb(os.path.join(img_path_prefix, p)) for p in item['image']], axis=0)
            obs_group.create_dataset("agentview_rgb", data=agentview_rgbs, compression="gzip")

            # 假设末端状态、夹爪状态、关节状态暂时用0填充（你可按实际补上）
            
            num_frames = len(item['image'])
            # obs_group.create_dataset("mode", data=mode, dtype=h5py.string_dtype(encoding='utf-8'))
            obs_group.create_dataset("ee_states", data=np.zeros((num_frames, 7), dtype=np.float32))  # pos(3) + quat(4)
            obs_group.create_dataset("gripper_states", data=np.zeros((num_frames, 1), dtype=np.float32))
            obs_group.create_dataset("joint_states", data=np.zeros((num_frames, 7), dtype=np.float32))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="data/franka/replay_traj_sampled_task_mv/real_robot_data_naive_single_input.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="data/franka/replay_traj_sampled_task_mv/real_robot_data_naive.hdf5", help="Output HDF5 path")
    args = parser.parse_args()
    main(args.jsonl, args.output)
