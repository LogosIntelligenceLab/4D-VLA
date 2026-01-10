# setsid nohup python tools/process_rlds.py > process_rlds_bridge.log 2>&1 &

import random
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm

import torch

import sys  # for debug
sys.path.append("/SSD_DISK/users/zhangjiahui/openvla/internvl_chat")
sys.path.append("/SSD_DISK/users/zhangjiahui/openvla")
from internvl.vla.datasets import RLDSDataset
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

    dataset = RLDSDataset(
        data_root_dir,
        data_mix,
        resize_resolution=default_image_resolution,
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
    )

    json_data1 = []
    json_data2 = []
    data_idx = 0
    dataset_length = dataset.dataset_length
    for rlds_batch in tqdm(dataset.dataset.as_numpy_iterator(), total=dataset_length, desc="Processing batches"):
        if data_idx == dataset_length:
            break
        dataset_name, action = rlds_batch["dataset_name"].decode(), rlds_batch["action"][0]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        img = rlds_batch["observation"]["image_primary"][0]
        img_wrist = rlds_batch["observation"]["image_wrist"][0]
        pil_img = Image.fromarray(img)
        pil_img_wrist = Image.fromarray(img_wrist)

        dataset_dir = os.path.join(data_root_dir, dataset_name, "images_v2_twoimage")
        os.makedirs(dataset_dir, exist_ok=True)

        image_path = f"{data_idx:08d}.jpg"
        image_path_wrist = f"{data_idx:08d}_wrist.jpg"
        # if not os.path.exists(image_path):
        pil_img.save(f"{dataset_dir}/{image_path}")
        pil_img_wrist.save(f"{dataset_dir}/{image_path_wrist}")
        # formatted_action = [f"{x:.4f}" for x in action]
        formatted_pos = [f"{round((x+1.)*5000)}" for x in action[:-1]]
        formatted_grip = [f"{round(action[-1]+1.)}"]
        formatted_action = formatted_pos + formatted_grip


        entry1 = {
            "id": data_idx,
            "image": image_path,
            "width": pil_img.width,
            "height": pil_img.height,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n[One image] " + f"What action should the robot take to {lang}"
                },
                {
                    "from": "gpt",
                    "value": f"<action>[{', '.join(formatted_action)}]</action>"

                }
            ]
        }

        entry2 = {
            "id": data_idx,
            "image": [image_path, image_path_wrist],
            "width": [pil_img.width, pil_img_wrist.width],
            "height": [pil_img.height, pil_img_wrist.height],
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n<image>\n[Two images] " + f"What action should the robot take to {lang}"
                },
                {
                    "from": "gpt",
                    "value": f"<action>[{', '.join(formatted_action)}]</action>"

                }
            ]
        }

        json_data1.append(entry1)
        json_data2.append(entry2)
        data_idx += 1

    len_data = len(json_data1) // 1000
    jsonl_file_name = f'{data_mix}_{len_data}k_v2_twoimage.jsonl'
    jsonl_file_path = os.path.join(data_root_dir, jsonl_file_name)

    with open(jsonl_file_path, "w") as jsonl_file:
        # for entry in json_data1:
        #     json.dump(entry, jsonl_file)
        #     jsonl_file.write('\n') 
        for entry in json_data2:
            json.dump(entry, jsonl_file)
            jsonl_file.write('\n') 
        

    print(f"Finished processing {data_idx} batches. Saved to {jsonl_file_path}")
        

    
