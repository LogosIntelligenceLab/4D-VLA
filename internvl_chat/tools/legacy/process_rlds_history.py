# setsid nohup python tools/process_rlds.py > process_rlds_bridge.log 2>&1 &

import random
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
from collections import deque

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
    data_mix = "libero_spatial_no_noops"
    default_image_resolution = (448, 448)
    shuffle_buffer_size = 0
    train = True
    image_aug = False

    history_len = 5  # HACK

    dataset = EpisodicRLDSDataset(
        data_root_dir,
        data_mix,
        resize_resolution=default_image_resolution,
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
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

        dataset_dir = os.path.join(data_root_dir, dataset_name, f"images_v2_{history_len}history")
        os.makedirs(dataset_dir, exist_ok=True)
        action_hist = deque(maxlen=history_len)
        image_path_hist = deque(maxlen=history_len)


        for fid in range(0, len(action_ep)):
            total_frames += 1
            
            pil_img = Image.fromarray(img_ep[fid])
            pil_img_wrist = Image.fromarray(img_wrist_ep[fid])

            image_path = f"ep{ep_idx}-fid{fid}.jpg"
            image_path_wrist = f"ep{ep_idx}-fid{fid}_wrist.jpg"

            pil_img.save(f"{dataset_dir}/{image_path}")
            pil_img_wrist.save(f"{dataset_dir}/{image_path_wrist}")

            # formatted_action = [f"{x:.4f}" for x in action]
            formatted_pos = [f"{round((x+1.)*5000)}" for x in action_ep[fid][:-1]]
            formatted_grip = [f"{round(action_ep[fid][-1]+1.)}"]
            formatted_action = formatted_pos + formatted_grip

            if len(action_hist) == 0:
                for _ in range(history_len-1):
                    action_hist.append(['5000', '5000', '5000', '5000', '5000', '5000', '1'])
                action_hist.append(formatted_action)
            else:
                action_hist.append(formatted_action)

            if len(image_path_hist) == 0:
                for _ in range(history_len):
                    image_path_hist.append(image_path)
            else:
                image_path_hist.append(image_path)

            entry1 = {
                "id": ep_idx,
                "image": [],
                "width_list": [],
                "height_list": [],
                "conversations": []
            }

            for i in range(history_len-1):
                    

                q = {
                    "from": "human",
                    "value": "<image>\n " + f"What action should the robot take to {lang}"
                }
                a = {
                    "from": "gpt",
                    "value": f"<action>[{', '.join(action_hist[i])}]</action>"
                }
                entry1["conversations"].append(q)
                entry1["conversations"].append(a)
                entry1['image'].append(image_path_hist[i])
                entry1['width_list'].append(448)
                entry1['height_list'].append(448)

            q = {
                "from": "human",
                "value": "<image>\n " + f"What action should the robot take to {lang}"
            }
            a = {
                "from": "gpt",
                "value": f"<action>[{', '.join(action_hist[-1])}]</action>"
            }
            entry1["conversations"].append(q)
            entry1["conversations"].append(a)
            entry1['image'].append(image_path_hist[-1])
            entry1['width_list'].append(448)
            entry1['height_list'].append(448)


            json_data1.append(entry1)

        ep_idx += 1

    len_data = len(json_data1) // 1000
    jsonl_file_name = f'{data_mix}_{len_data}k_v2_{history_len}history.jsonl'
    jsonl_file_path = os.path.join(data_root_dir, jsonl_file_name)

    with open(jsonl_file_path, "w") as jsonl_file:
        for entry in json_data1:
            json.dump(entry, jsonl_file)
            jsonl_file.write('\n') 
        # for entry in json_data2:
        #     json.dump(entry, jsonl_file)
        #     jsonl_file.write('\n') 
        

    print(f"Finished processing {total_frames} batches. Saved to {jsonl_file_path}")
        

    
