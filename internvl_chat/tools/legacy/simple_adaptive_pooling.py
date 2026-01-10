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
from internvl.model.internvl_chat.adaptive_pooling import DINOv2FeatureExtractor, temporal_reduction_dino, anchor_window_dedup
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
    # data_mix = "libero_spatial_no_noops_3d_change_pose_large_uniform"
    # data_mix = "libero_10_no_noops_3d"
    data_mix = "libero_spatial_no_noops_3d"
    default_image_resolution = (448, 448)
    shuffle_buffer_size = 0
    train = True
    image_aug = False

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
    dino_extractor = DINOv2FeatureExtractor(device="cuda", model_name="vit_base_patch14_dinov2", img_size=518)
    json_data1 = []
    json_data2 = []
    ep_idx = 0
    total_frames = 0
    dataset_length = dataset.dataset_length
    id_list_cnt = []
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

        camera_info_agentview_extrinsic = rlds_batch["camera_info"]['camera_info_agentview_extrinsic']
        camera_info_robothand_extrinsic = rlds_batch["camera_info"]['camera_info_robothand_extrinsic']
        

        camera_info_fovy = rlds_batch["camera_info"]['camera_info_fovy']


        dataset_dir = os.path.join(data_root_dir, dataset_name, f"adaptive_pooling")
        os.makedirs(dataset_dir, exist_ok=True)
        image_path_hist = []
        image_feats_hist = []
        id_hist = []

        is_zero_saved = False
        
        with torch.no_grad():
            img_feat_ep = dino_extractor.extract(img_ep) # img_ep.shape: (T, H, W, 3), img_feat_ep.shape: (T, 768)
        # memory_bank.feature_extractor.resnet50.to("cuda")
        # img_feat_ep = memory_bank.feature_extractor.extract_image_feature(img_ep)
        
        for fid in range(0, len(action_ep)):
            total_frames += 1
            
            pil_img = Image.fromarray(img_ep[fid])
            pil_img_wrist = Image.fromarray(img_wrist_ep[fid])

            pil_zero = Image.fromarray(np.zeros_like(img_ep[fid]))

            image_path = f"ep{ep_idx}-fid{fid}.jpg"


            pil_img.save(f"{dataset_dir}/{image_path}")
            
            if not is_zero_saved:
                pil_zero = Image.fromarray(np.zeros_like(img_ep[fid]))
                image_path_zero = "zero.jpg"
                pil_zero.save(f"{dataset_dir}/{image_path_zero}")
                is_zero_saved = True

            if len(image_path_hist) == 0:
                # for _ in range(history_len):
                #     image_path_hist.append(image_path_zero)
                #     image_feats_hist.append(torch.zeros_like(img_feat_ep[fid]))  # 假设特征维度为768

                image_path_hist.append(image_path)
                image_feats_hist.append(img_feat_ep[fid])
                id_hist.append(fid)
            else:
                image_path_hist.append(image_path)
                image_feats_hist.append(img_feat_ep[fid])
                id_hist.append(fid)
            
            # if len(image_path_hist) >= 20:
            #     temporal_keep = temporal_reduction_dino(image_feats_hist, window_size=20, sim_thresh=0.99)
            #     selected_fids = anchor_window_dedup(image_feats_hist, temporal_keep, window_size=20, anchor_thresh=0.98)
            # else:
            #     temporal_keep = torch.ones(len(image_path_hist), dtype=torch.bool)
            #     selected_fids = anchor_window_dedup(image_feats_hist, temporal_keep, window_size=20, anchor_thresh=0.98)
            #     selected_fids = list(range(len(image_path_hist)))
            temporal_keep = temporal_reduction_dino(image_feats_hist, window_size=20, sim_thresh=0.99)
            selected_fids = anchor_window_dedup(image_feats_hist, temporal_keep, window_size=20, anchor_thresh=0.96)
            id_list_cnt.append(len(selected_fids))
            rounded_action = [round(float(x), 5) for x in action_ep[fid]]
            entry1 = {
                "id": total_frames,
                "image": [],
                "image_depth": [],
                "height_list": [],
                "width_list": [],
                "conversations": [],
                "action": rounded_action, 
                
            }

            for f in selected_fids:
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
                "value": "History:\n:" + "<image>\n" * (len(entry1['image'])-1) +  "Current:\n<image>\n " + f"What action should the robot take to {lang}"
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
    jsonl_file_name = f'{data_mix}_{len_data}k_v2_adaptivate_pooling.jsonl'
    jsonl_file_path = os.path.join(data_root_dir, jsonl_file_name)

    with open(jsonl_file_path, "w") as jsonl_file:
        for entry in json_data1:
            json.dump(entry, jsonl_file)
            jsonl_file.write('\n') 


    print(f"Finished processing {total_frames} batches. Saved to {jsonl_file_path}")
        

    
