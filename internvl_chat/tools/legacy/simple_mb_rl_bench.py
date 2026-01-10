# setsid nohup python tools/process_rlds.py > process_rlds_bridge.log 2>&1 &

import random
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
from collections import deque
import cv2
import pickle
import torch
from scipy.spatial.transform import Rotation as R

import sys  # for debug
sys.path.append("/SSD_DISK/users/zhangjiahui/openvla/internvl_chat")
sys.path.append("/SSD_DISK/users/zhangjiahui/openvla")
from internvl.model.internvl_chat.memory_bank_exp2 import SimpleFeatureExtractor, SimpleMemoryBank

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def get_curr_pos(item):
    gripper_pos = getattr(item, "gripper_pose")
    gripper_open = getattr(item, "gripper_open")
    quat_rot = gripper_pos[-4:]

    euler = R.from_quat(quat_rot).as_euler("XYZ")
    gripper_position = np.array([gripper_open], dtype=np.float32).reshape(-1, 1)


    # here calculate the action for action head return
    # rot = quat_to_rot_6d(quat_rot.reshape(1, -1)) # contruct the 6d rot
    cartesian_position = np.concatenate([gripper_pos[:3].reshape(1, -1), euler.reshape(1, -1)], axis=1, dtype=np.float32)
    return cartesian_position, gripper_position


def compute_delta_action(cartesian_position, cartesian_position_cmd,
                         euler_seq: str = 'xyz', normalize: bool = True):
    """
    计算 delta action（位置差 + 欧拉角差，欧拉角单位：度）
    Args:
        cartesian_position:      [x, y, z, e1, e2, e3]  当前位姿（欧拉角单位：度）
        cartesian_position_cmd:  [x, y, z, e1, e2, e3]  目标位姿（欧拉角单位：度）
        euler_seq: 欧拉角顺序（如 'xyz', 'zyx' 等），需与输入角一致
        normalize: 是否将欧拉角差规范到 (-180, 180]
    Returns:
        np.ndarray(6,) = [dx, dy, dz, de1_deg, de2_deg, de3_deg]
    """
    cur = np.asarray(cartesian_position, dtype=float).reshape(6)
    cmd = np.asarray(cartesian_position_cmd, dtype=float).reshape(6)

    # 1) 位置差
    dpos = cmd[:3] - cur[:3]

    # 2) 姿态差：欧拉(度) -> 旋转 -> 相对旋转 -> 欧拉(度)
    R_cur = R.from_euler(euler_seq, cur[3:], degrees=False)
    R_cmd = R.from_euler(euler_seq, cmd[3:], degrees=False)
    R_delta = R_cmd * R_cur.inv()
    deul = R_delta.as_euler(euler_seq, degrees=False)

    if normalize:
        deul = (deul + np.pi) % (2 * np.pi) - np.pi 

    return np.concatenate([dpos, deul])

if __name__ == "__main__":
    default_image_resolution = (448, 448)
    shuffle_buffer_size = 0
    train = True
    image_aug = False


    window_size = 20
    bank_size = 5
    history_len = bank_size
    sample_interval = 1


    memory_bank = SimpleMemoryBank(window_size=window_size, bank_size=bank_size)

    json_data1 = []
    json_data2 = []
    ep_idx = 0
    total_frames = 0
    
    # task_name = "meat_off_grill"
    task_name = "push_buttons"
    # task_name = "place_wine_at_rack_location"
    data_folder = f"/SSD_DISK/users/zhangjiahui/arm4r/sim/data/train/{task_name}/all_variations/episodes"
    data_eposide_list = sorted(os.listdir(data_folder))
    # for ep_idx, epi_name in enumerate(data_eposide_list):
    for ep_idx, epi_name in enumerate(tqdm(data_eposide_list, desc="Processing Episodes")):

        variation_des_pickle = os.path.join(data_folder, epi_name, "variation_descriptions.pkl")
        with open(variation_des_pickle, 'rb') as file:
            variation_des = pickle.load(file)
        lang = variation_des[0]
        pickle_path = os.path.join(data_folder, epi_name, "low_dim_obs.pkl")

        with open(pickle_path, 'rb') as file:
            data = pickle.load(file)
        # print(f"Processing episode: {epi_name}")
        cartesian_positions = []
        gripper_positions = []
        cartesian_positions_cmd = []
        gripper_positions_cmd = []
        exterior_image_1_left = []
        wrist_image_left = []
        img1_test = []
        img2_test = []


        dataset_dir = f"/SSD_DISK/users/zhangjiahui/openvla/data/rlbench/{task_name}"
        os.makedirs(dataset_dir, exist_ok=True)
        action_hist = deque(maxlen=history_len)
        image_path_hist = deque(maxlen=history_len)
        image_wrist_hist = deque(maxlen=history_len)
        image_depth_path_hist = deque(maxlen=history_len)

        is_zero_saved = False
        memory_bank.feature_extractor.resnet50.to("cuda")

        # img_feat_ep = memory_bank.feature_extractor.extract_image_feature(img_ep)

        memory_bank.clear()
        current_episode_path = os.path.join(data_folder, epi_name)
        for fid in range(0, len(data)-1):
            total_frames += 1
            
            exterior_image_1_left_path = '{}/{}/{}.png'.format(current_episode_path, "front_rgb", fid)
            # try:
            #     img1 = Image.open(exterior_image_1_left_path)
            # except:
            #     img1 = np.zeros([448,448,3])
            pil_img = Image.open(exterior_image_1_left_path)
            wrist_image_left_path = '{}/{}/{}.png'.format(current_episode_path, "wrist_rgb", fid)
            # try:
            #     img2 = Image.open(wrist_image_left_path)
            # except:
            #     img2 = np.zeros([448, 448, 3])
            pil_img_wrist = Image.open(wrist_image_left_path)

            image_path = f"ep{ep_idx}-fid{fid}.jpg"
            image_path_wrist = f"ep{ep_idx}-fid{fid}_wrist.jpg"

            pil_img.save(f"{dataset_dir}/{image_path}")
            pil_img_wrist.save(f"{dataset_dir}/{image_path_wrist}")

            if not is_zero_saved:
                pil_zero = Image.fromarray(np.zeros_like(np.array(pil_img)))
                image_path_zero = "zero.jpg"
                pil_zero.save(f"{dataset_dir}/{image_path_zero}")
                is_zero_saved = True
            weist_np = np.array(pil_img_wrist)[None]
            wrist_feats = memory_bank.feature_extractor.extract_image_feature(weist_np)[0]
            if len(image_path_hist) == 0:
                for _ in range(history_len):
                    image_path_hist.append(image_path_zero)
                    image_wrist_hist.append(image_path_zero)
                    memory_bank.add(wrist_feats, fid, from_img=False)

                image_path_hist.append(image_path)
                image_wrist_hist.append(image_path_wrist)
                memory_bank.add(wrist_feats, fid, from_img=False)
            else:
                image_path_hist.append(image_path)
                image_wrist_hist.append(image_path_wrist)
                memory_bank.add(wrist_feats, fid, from_img=False)
            
            cartesian_position, gripper_position = get_curr_pos(data[fid])
            cartesian_position_cmd, gripper_position_cmd = get_curr_pos(data[fid + 1])
            delt_action = compute_delta_action(cartesian_position, cartesian_position_cmd)
            
            action_ep = np.concatenate([delt_action, gripper_position_cmd[0]]).reshape(-1)

            rounded_action = [round(float(x), 5) for x in action_ep]
            entry1 = {
                "id": total_frames,
                "image": [],
                "image_depth": [],
                "width_list": [],
                "height_list": [],
                "conversations": [],
                "action": rounded_action, 
            }

            fid_list = memory_bank.get_fid()[:-1]
            for f in fid_list:
                img_path = f"ep{ep_idx}-fid{f}.jpg"
                entry1['image'].append(img_path)
                entry1['width_list'].append(448)
                entry1['height_list'].append(448)

            img_path = f"ep{ep_idx}-fid{fid}.jpg"
            entry1['image'].append(img_path)
            entry1['width_list'].append(448)
            entry1['height_list'].append(448)

            q = {
                "from": "human",
                "value": "History:\n:" + "<image>\n" * (history_len//sample_interval-1) +  "Current:\n<image>\n<image>\n " + f"What action should the robot take to {lang}"
            }
            a = {
                "from": "gpt",
                "value": f"<state_pred>"
            }
            entry1["conversations"].append(q)
            entry1["conversations"].append(a)


            json_data1.append(entry1)

        ep_idx += 1

    len_data = len(json_data1) // 1000
    jsonl_file_name = f'{task_name}_{len_data}k_.jsonl'
    data_root_dir = "/SSD_DISK/users/zhangjiahui/openvla/data/rlbench"
    jsonl_file_path = os.path.join(data_root_dir, jsonl_file_name)

    with open(jsonl_file_path, "w") as jsonl_file:
        for entry in json_data1:
            json.dump(entry, jsonl_file)
            jsonl_file.write('\n') 

    
    print(f"Finished processing {total_frames} batches. Saved to {jsonl_file_path}")
        

    
