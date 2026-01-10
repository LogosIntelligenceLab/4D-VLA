# setsid nohup python tools/process_rlds.py > process_rlds_bridge.log 2>&1 &

import random
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
from collections import deque
import cv2
import glob
import torch
from scipy.spatial.transform import Rotation as R

import sys  # for debug
sys.path.append("/SSD_DISK/users/zhangjiahui/openvla/internvl_chat")
sys.path.append("/SSD_DISK/users/zhangjiahui/openvla")
# from internvl.vla.datasets import RLDSDataset, EpisodicRLDSDataset
from internvl.model.internvl_chat.memory_bank_exp2 import SimpleFeatureExtractor, SimpleMemoryBank
# from prismatic.vla.datasets import RLDSDataset

def quat_to_matrix(q):
    """将四元数转为 3x3 旋转矩阵"""
    return R.from_quat(q).as_matrix()

def delta_rot_quat(prev_q, curr_q):
    """
    计算两个四元数之间的旋转差，并转为 XYZ 欧拉角（弧度）
    - 输入: prev_q, curr_q: 四元数 [x, y, z, w]
    - 输出: delta_euler: [rx, ry, rz]，单位弧度
    """
    R_prev = R.from_quat(prev_q).as_matrix()
    R_curr = R.from_quat(curr_q).as_matrix()
    
    R_delta = R_prev.T @ R_curr
    delta_euler = R.from_matrix(R_delta).as_euler('xyz', degrees=False)
    return delta_euler.tolist()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
prompt_mapping = {
    "rgb": "Please place the red, green, and blue blocks into the basket in order.",
    "rbg": "Please place the red, blue, and green blocks into the basket in order.",
    "grb": "Please place the green, red, and blue blocks into the basket in order.",
    "gbr": "Please place the green, blue, and red blocks into the basket in order.",
    "brg": "Please place the blue, red, and green blocks into the basket in order.",
    "bgr": "Please place the blue, green, and red blocks into the basket in order.",
}

if __name__ == "__main__":
    whole_data_root_dir = "/SSD_DISK/users/zhangjiahui/openvla/data/franka/replay_traj_sampled_task5"
    default_image_resolution = (448, 448)
    shuffle_buffer_size = 0
    train = True
    image_aug = False


    window_size = 20
    bank_size = 5
    history_len = bank_size
    sample_interval = 1



    
    memory_bank = SimpleMemoryBank(window_size=window_size, bank_size=bank_size)
    memory_bank.feature_extractor.resnet50.to("cuda")
    color_folders = os.listdir(whole_data_root_dir)
    for color_folder in color_folders:
        mode = color_folder
        if "zip" in color_folder:
            continue
        if "delete" in color_folder:
            continue
        if "json" in color_folder:
            continue
        data_root_dir = os.path.join(whole_data_root_dir, color_folder)
                                              
        data_folders = os.listdir(data_root_dir)
        for data_folder in data_folders:
            if not os.path.isdir(os.path.join(data_root_dir, data_folder)):
                continue
            json_files = glob.glob(os.path.join(data_root_dir, data_folder, "*.json"))[0]
            
            with open(json_files, "r") as f:
                data = json.load(f)
                
            prev_pos, prev_quat = None, None

            is_zero_saved = False
            total_frames = 0
            json_data1 = []
            image_path_hist = deque(maxlen=history_len)
            depth_path_hist = deque(maxlen=history_len)
            proprio_hist = deque(maxlen=history_len)
            memory_bank.clear()
            
            for fid, item in enumerate(tqdm(data)):
                if fid == len(data)-1:  # check
                    # prev_pos = np.array(item["executed_ee_position"])
                    # prev_quat = np.array(item["executed_ee_quaternion"])
                    # prev_quat = [prev_quat[1], prev_quat[2], prev_quat[3], prev_quat[0]]
                    continue
                total_frames += 1
                os.makedirs(os.path.join(data_root_dir, data_folder, "images_processed"), exist_ok=True)
                img_path = os.path.join(data_root_dir, data_folder, "images", item["color_image"])
                depth_npy_path = os.path.join(data_root_dir, data_folder, "images", item["depth_npy"])
                
                pil_img = Image.open(img_path).convert("RGB")
                depth_arr = np.load(depth_npy_path)  # unit?
                
                rgb_name = f"fid{fid}.png"
                depth_name = f"fid{fid}_depth.npy"
                if not os.path.exists(os.path.join(data_root_dir, data_folder, "images_processed", rgb_name)):
                    pil_img.save(os.path.join(data_root_dir, data_folder, "images_processed", rgb_name))
                    np.save(os.path.join(data_root_dir, data_folder, "images_processed", depth_name), depth_arr)
                    
                    if not is_zero_saved:
                        Image.fromarray(np.zeros_like(np.array(pil_img))).save(os.path.join(data_root_dir, data_folder, "images_processed", "zero.jpg"))
                        np.save(os.path.join(data_root_dir, data_folder, "images_processed", "zero_depth.npy"), np.zeros_like(depth_arr))
                        is_zero_saved = True
                    
                
                if len(image_path_hist) == 0:
                    for _ in range(history_len):
                        memory_bank.add(np.array(pil_img)[None], fid, real=True)
                    memory_bank.add(np.array(pil_img)[None], fid, real=True)
                else:
                    memory_bank.add(np.array(pil_img)[None], fid, real=True)

                curr_pos = np.array(item["executed_ee_position"])
                curr_quat = np.array(item["executed_ee_quaternion"])
                # curr_quat = [curr_quat[1], curr_quat[2], curr_quat[3], curr_quat[0]]  # 转为 [x, y, z, w]
                
                tar_pos = np.array(data[fid+1]["executed_ee_position"])
                tar_quat = np.array(data[fid+1]["executed_ee_quaternion"])
                # tar_quat = [tar_quat[1], tar_quat[2], tar_quat[3], tar_quat[0]]  # 转为 [x, y, z, w]

                delta_pos = (tar_pos - curr_pos).tolist()
                delta_rot = delta_rot_quat(curr_quat, tar_quat)

                if data[fid+1]["gripper_width"] > 0.07:
                    gripper_width = 1
                else:
                    gripper_width = 0
                    
                action = delta_pos + delta_rot + [gripper_width]
                action = [round(a, 5) for a in action]
                intrinsic = [606.359375, 389.8868408203125, 320.4429626464844, 240.7040252685547, 640, 480]
                fx, fy, cx, cy, w, h = intrinsic

                raw_size = (w, h)
                target_size = (448, 448)
                scale_x = target_size[0] / raw_size[0]
                scale_y = target_size[1] / raw_size[1]

                agentview_intrinsic = [
                    fx * scale_x, 0,         cx * scale_x,
                    0,           fy * scale_y, cy * scale_y,
                    0,           0,         1
                ]
                agentview_intrinsic = np.array(agentview_intrinsic).reshape(3, 3)
                agentview_intrinsic = np.round(agentview_intrinsic, 4)
                
                quat = [-0.6579468466651127, -0.6194541929018136, 0.28531243677131374, 0.31934192221412583]  # w, x, y, z
                trans = [1.187292437489465, -0.09629429279576884, 0.964384707990467]
                rot_mat = R.from_quat(quat).as_matrix()  # shape: [3, 3]

                agentview_extrinsic = np.eye(4)
                agentview_extrinsic[:3, :3] = rot_mat
                agentview_extrinsic[:3, 3] = trans
                agentview_extrinsic = np.round(agentview_extrinsic, 4)
                
                agentview_extrinsic_list = agentview_extrinsic.tolist()
                agentview_intrinsic_list = agentview_intrinsic.tolist()
                # lang = "move the yellow block to the plate"  # TODO: replace with actual language
                # lang = "put all the green cubes on the table into the plate"
                # lang = "put the yellow block on top of the red block"
                lang = prompt_mapping[mode]
                entry = {
                    "id": total_frames,
                    "mode": mode,
                    "image": [],
                    "image_depth": [],
                    "width_list": [],
                    "height_list": [],
                    "width_depth_list": [],
                    "height_depth_list": [],
                    "action": action,
                    "camera_info": {
                        "agentview_extrinsic": agentview_extrinsic_list,
                        "agentview_intrinsic": agentview_intrinsic_list,
                    },
                    
                    
                    "conversations": [
                        {
                            "from": "human",
                            "value": "History:\n" + "<image>\n" * (history_len - 1) + f"Current:\n<image>\n What action should the robot take to {lang}?"
                        },
                        {
                            "from": "gpt",
                            "value": "<state_pred>"
                        }
                    ]
                }

                fid_list = memory_bank.get_fid()[:-1]
                # for i in fid_list:
                #     f = fid_list[-1]
                #     entry["image"].append(f"{data_folder}/images_processed/{rgb_name}")
                #     entry["image_depth"].append(f"{data_folder}/images_processed/{depth_name}")
                #     entry["width_list"].append(448)
                #     entry["height_list"].append(448)
                #     entry["width_depth_list"].append(256)
                #     entry["height_depth_list"].append(256)

                entry["image"].append(os.path.join(data_folder, "images_processed", rgb_name))
                entry["image_depth"].append(os.path.join(data_folder, "images_processed", depth_name))
                entry["width_list"].append(448)
                entry["height_list"].append(448)
                entry["width_depth_list"].append(256)
                entry["height_depth_list"].append(256)

                json_data1.append(entry)
                
            output_jsonl = os.path.join(data_root_dir, data_folder, "real_robot_data_naive_single_input.jsonl".format(window_size, bank_size))
            with open(output_jsonl, "w") as f:
                for entry in json_data1:
                    json.dump(entry, f)
                    f.write("\n")
            print(f"Saved {len(json_data1)} samples to {output_jsonl}")

    jsonl_path = []
    for color in color_folders:
        if "zip" in color:
            continue
        if "delete" in color:
            continue
        if "json" in color:
            continue
        data_root_dir = os.path.join(whole_data_root_dir, color)
        data_folders = os.listdir(data_root_dir)
        for data_folder in data_folders:
            if not os.path.isdir(os.path.join(data_root_dir, data_folder)):
                continue
            jsonl_path.append(os.path.join(data_root_dir, data_folder, "real_robot_data_naive_single_input.jsonl"))
    
    final_jsonl_path = os.path.join(whole_data_root_dir, "real_robot_data_naive_single_input.jsonl")
    # jsonl_path = [os.path.join(data_root_dir, data_folder, "real_robot_data_naive_single_input.jsonl".format(window_size, bank_size)) for data_folder in data_folders if os.path.isdir(os.path.join(data_root_dir, data_folder))]
    # final_jsonl_path = os.path.join(data_root_dir, "real_robot_data_naive_single_input.jsonl".format(window_size, bank_size))
    with open(final_jsonl_path, "w") as outfile:
        for jsonl in jsonl_path:
            with open(jsonl, "r") as infile:
                for line in infile:
                    outfile.write(line)
    print(f"Final jsonl saved to {final_jsonl_path}")