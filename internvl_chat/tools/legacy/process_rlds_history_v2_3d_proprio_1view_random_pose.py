# setsid nohup python tools/process_rlds.py > process_rlds_bridge.log 2>&1 &

import random
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
from collections import deque
import cv2

import torch

import sys  # for debug
sys.path.append("/SSD_DISK/users/zhangjiahui/openvla/internvl_chat")
sys.path.append("/SSD_DISK/users/zhangjiahui/openvla")
from internvl.vla.datasets import RLDSDataset, EpisodicRLDSDataset
# from prismatic.vla.datasets import RLDSDataset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

if __name__ == "__main__":
    data_root_dir = "../data/"
    # data_mix = "libero_spatial_no_noops_3d_change_pose_large"
    data_mix = "libero_spatial_no_noops_3d_change_pose_large_uniform"
    default_image_resolution = (448, 448)
    shuffle_buffer_size = 0
    train = True
    image_aug = False
    sample_interval = 1  # HACK

    history_len = 1  # HACK

    dataset = EpisodicRLDSDataset(
        data_root_dir,
        data_mix,
        resize_resolution=default_image_resolution,
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        load_depth=True,
        load_camera_info=True,
    )

    json_data1 = []
    json_data2 = []
    ep_idx = 0
    total_frames = 0
    dataset_length = dataset.dataset_length

    for rlds_batch in tqdm(dataset.dataset.as_numpy_iterator(), total=dataset_length, desc="Processing batches"):
        if ep_idx == dataset_length:
        # if ep_idx == 5:
            break
        dataset_name, action_ep = rlds_batch["dataset_name"][0].decode(), rlds_batch["action"][:, 0]
        lang = rlds_batch["task"]["language_instruction"][0].decode().lower()
        img_ep = rlds_batch["observation"]["image_primary"][:, 0]
        img_wrist_ep = rlds_batch["observation"]["image_wrist"][:, 0]
        img_ep_d = rlds_batch['depth']['depth_primary']
        img_wrist_ep_d = rlds_batch['depth']['depth_wrist']
        proprio_ep = rlds_batch["observation"]['proprio']

        # camera_info_pos = rlds_batch["camera_info"]['camera_info_position']
        # camera_info_quat = rlds_batch["camera_info"]['camera_info_quaternion']
        # camera_info_intrinsic = rlds_batch["camera_info"]['camera_info_intrinsic']

        camera_info_agentview_extrinsic = rlds_batch["camera_info"]['camera_info_agentview_extrinsic']
        camera_info_robothand_extrinsic = rlds_batch["camera_info"]['camera_info_robothand_extrinsic']
        

        camera_info_fovy = rlds_batch["camera_info"]['camera_info_fovy']

        translation_range = 0.5
        random_translation = np.random.uniform(-translation_range, translation_range, 3)

        dataset_dir = os.path.join(data_root_dir, dataset_name, f"images_v2_{history_len}history-v2-{sample_interval}sample_interval_3d_proprio_1view_random_pose{translation_range}")
        os.makedirs(dataset_dir, exist_ok=True)
        action_hist = deque(maxlen=history_len)
        image_path_hist = deque(maxlen=history_len)
        image_depth_path_hist = deque(maxlen=history_len)
        proprio_hist = deque(maxlen=history_len)

        is_zero_saved = False



        for fid in range(0, len(action_ep)):
            total_frames += 1
            
            pil_img = Image.fromarray(img_ep[fid])
            pil_img_wrist = Image.fromarray(img_wrist_ep[fid])

            # pil_img_d = Image.fromarray(img_ep_d[fid])
            # pil_img_wrist_d = Image.fromarray(img_wrist_ep_d[fid])

            pil_zero = Image.fromarray(np.zeros_like(img_ep[fid]))

            image_path = f"ep{ep_idx}-fid{fid}.jpg"
            image_path_wrist = f"ep{ep_idx}-fid{fid}_wrist.jpg"
            image_depth_path = f"ep{ep_idx}-fid{fid}_depth.png"
            image_depth_wrist_path = f"ep{ep_idx}-fid{fid}_depth_wrist.png"

            pil_img.save(f"{dataset_dir}/{image_path}")
            pil_img_wrist.save(f"{dataset_dir}/{image_path_wrist}")
            # pil_img_d.save(f"{dataset_dir}/{image_depth_path}")
            # pil_img_wrist_d.save(f"{dataset_dir}/{image_depth_wrist_path}")
            cv2.imwrite(f"{dataset_dir}/{image_depth_path}", img_ep_d[fid])
            cv2.imwrite(f"{dataset_dir}/{image_depth_wrist_path}", img_wrist_ep_d[fid])
            
            if not is_zero_saved:
                pil_zero = Image.fromarray(np.zeros_like(img_ep[fid]))
                image_path_zero = "zero.jpg"
                pil_zero.save(f"{dataset_dir}/{image_path_zero}")
                is_zero_saved = True

            # formatted_action = [f"{x:.4f}" for x in action]
            # formatted_pos = [f"{round((x+1.)*5000)}" for x in action_ep[fid][:-1]]
            # formatted_grip = [f"{round(action_ep[fid][-1]+1.)}"]
            # formatted_action = formatted_pos + formatted_grip

            # if len(action_hist) == 0:
            #     for _ in range(history_len-1):
            #         action_hist.append(['5000', '5000', '5000', '5000', '5000', '5000', '1'])
            #     action_hist.append(formatted_action)
            # else:
            #     action_hist.append(formatted_action)

            if len(image_path_hist) == 0:
                for _ in range(history_len):
                    image_path_hist.append(image_path_zero)
                    image_depth_path_hist.append(image_path_zero)
                image_path_hist.append(image_path)
                image_depth_path_hist.append(image_depth_path)
            else:
                image_path_hist.append(image_path)
                image_depth_path_hist.append(image_depth_path)

            if len(proprio_hist) == 0:
                for _ in range(history_len):
                    proprio_hist.append(np.zeros_like(proprio_ep[fid]))
                proprio_hist.append(proprio_ep[fid])
            else:
                proprio_hist.append(proprio_ep[fid])

            
            rounded_action = [round(float(x), 5) for x in action_ep[fid]]


            agentview_extrinsic = camera_info_agentview_extrinsic[fid].copy()
            agentview_extrinsic[:3, 3] += random_translation

            robothand_extrinsic = camera_info_robothand_extrinsic[fid].copy()
            robothand_extrinsic[:3, 3] += random_translation

            entry1 = {
                "id": total_frames,
                "image": [],
                "image_depth": [],
                "width_list": [],
                "height_list": [],
                "width_depth_list": [],
                "height_depth_list": [],
                "conversations": [],
                "proprio": [],
                "action": rounded_action, 
                
                "camera_info": {
                    # "position": camera_info_pos[fid].tolist(),
                    # "quaternion": camera_info_quat[fid].tolist(),
                    # "intrinsic": camera_info_intrinsic[fid].tolist(),
                    "agentview_extrinsic": agentview_extrinsic.tolist(),
                    "robothand_extrinsic": robothand_extrinsic.tolist(),
                    "fovy": camera_info_fovy[fid].tolist()
                }
            }

            for i in range(history_len):
                if (i+1) % sample_interval == 0:
                    entry1['image'].append(image_path_hist[i])
                    entry1['width_list'].append(448)
                    entry1['height_list'].append(448)
                    entry1['image_depth'].append(image_depth_path_hist[i])
                    entry1['width_depth_list'].append(256)
                    entry1['height_depth_list'].append(256)
                    proprio = proprio_hist[i][0].copy()
                    proprio[:3] += random_translation
                    proprio[3:] = np.zeros_like(proprio[3:])
                    entry1['proprio'].append(proprio.tolist())

    
                

            q = {
                "from": "human",
                # "value": "History:\n:" + "<image>\n" * (history_len//sample_interval-1) +  "Current:\n<image>\n " + f"What action should the robot take to {lang}"
                # "value": "History:\n:" + "<proprio>\n" +  "Current:\n<image>\n " + f"What action should the robot take to {lang}"
                "value": "Current:\n<image>\n " + f"What action should the robot take to {lang}"
            }
            a = {
                "from": "gpt",
                "value": f"<state_pred>"
                # "value": f"<action>[{', '.join(formatted_action)}]</action>"
            }
            entry1["conversations"].append(q)
            entry1["conversations"].append(a)


            json_data1.append(entry1)

        ep_idx += 1

    len_data = len(json_data1) // 1000
    jsonl_file_name = f'{data_mix}_{len_data}k_v2_{history_len}history-v2-{sample_interval}sample_interval_3d_proprio_1view_random_pose{translation_range}.jsonl'
    jsonl_file_path = os.path.join(data_root_dir, jsonl_file_name)

    with open(jsonl_file_path, "w") as jsonl_file:
        for entry in json_data1:
            json.dump(entry, jsonl_file)
            jsonl_file.write('\n') 
        # for entry in json_data2:
        #     json.dump(entry, jsonl_file)
        #     jsonl_file.write('\n') 
        

    print(f"Finished processing {total_frames} batches. Saved to {jsonl_file_path}")
        

    
