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
from transformers import BertTokenizer, BertModel
import zarr

import sys  # for debug
sys.path.append("/SSD_DISK/users/zhangjiahui/openvla/internvl_chat")
sys.path.append("/SSD_DISK/users/zhangjiahui/openvla")
from internvl.vla.datasets import RLDSDataset, EpisodicRLDSDataset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
import open_clip

class Text_Tokenizer():
    def __init__(self):
        self.model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def tokenize(self, text):
        # image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
        text = self.tokenizer([text])

        with torch.no_grad(), torch.cuda.amp.autocast():
            # image_features = model.encode_image(image)
            text_features = self.model.encode_text(text)

        return text_features

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

if __name__ == "__main__":
    data_root_dir = "/SSD_disk/users/zhangjiahui/openvla/data"
    # data_mix = "libero_spatial_no_noops_3d_change_pose_large"
    # data_mix = "libero_spatial_no_noops_3d_change_pose_large_uniform"
    data_mix = "libero_spatial_no_noops_3d"
    name = data_mix.split("_")[1]
    if name == '10':
        name = 'long'
    default_image_resolution = (448, 448)
    shuffle_buffer_size = 0
    train = True
    image_aug = False
    sample_interval = 1  # HACK

    history_len = 5  # HACK

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
    text_tokenizer = Text_Tokenizer()

    json_data1 = []
    json_data2 = []
    ep_idx = 0
    total_frames = 0
    dataset_length = dataset.dataset_length
    instruction_embedding_dict = {}
    for rlds_batch in tqdm(dataset.dataset.as_numpy_iterator(), total=dataset_length, desc="Processing batches"):
        if ep_idx == dataset_length:
            break
        dataset_name, action_ep = rlds_batch["dataset_name"][0].decode(), rlds_batch["action"][:, 0]
        lang = rlds_batch["task"]["language_instruction"][0].decode().lower()
        img_ep = rlds_batch["observation"]["image_primary"][:, 0]
        img_wrist_ep = rlds_batch["observation"]["image_wrist"][:, 0]
        img_ep_d = rlds_batch['depth']['depth_primary']
        img_wrist_ep_d = rlds_batch['depth']['depth_wrist']
        proprio_ep = rlds_batch["observation"]['proprio']
        # if lang != "pick up the black bowl between the plate and the ramekin and place it on the plate":
        #     continue

        # camera_info_pos = rlds_batch["camera_info"]['camera_info_position']
        # camera_info_quat = rlds_batch["camera_info"]['camera_info_quaternion']
        # camera_info_intrinsic = rlds_batch["camera_info"]['camera_info_intrinsic']

        camera_info_agentview_extrinsic = rlds_batch["camera_info"]['camera_info_agentview_extrinsic']
        camera_info_robothand_extrinsic = rlds_batch["camera_info"]['camera_info_robothand_extrinsic']
        camera_info_fovy = rlds_batch["camera_info"]['camera_info_fovy']


        dataset_dir = os.path.join(data_root_dir, dataset_name, f"images_arm4r_{history_len}history-v2-{sample_interval}sample_interval_3d_proprio_1view")
        os.makedirs(dataset_dir, exist_ok=True)
        action_hist = deque(maxlen=history_len)
        image_path_hist = deque(maxlen=history_len)
        image_depth_path_hist = deque(maxlen=history_len)
        proprio_hist = deque(maxlen=history_len)

        eps_dir = os.path.join(f"/SSD_disk/users/zhangjiahui/openvla/data/anns_{name}_arm4r_unscale/task1", f"kinova_tasks/common_task/eps{ep_idx}")
        os.makedirs(eps_dir, exist_ok=True)
        # instruction_embedding = get_bert_embedding(f"What action should the robot take to {lang}")
        # import pdb; pdb.set_trace()
        if lang not in instruction_embedding_dict:
            instruction_embedding_dict[lang] = text_tokenizer.tokenize(f"What action should the robot take to {lang}").cpu().numpy()[0] 
            # instruction_embedding_dict[lang] = get_bert_embedding(f"What action should the robot take to {lang}")
        instruction_embedding = instruction_embedding_dict[lang]
        # instruction_embedding = instruction_embedding_dict.get(
        #     lang, get_bert_embedding(f"What action should the robot take to {lang}")
        # )

        with open(os.path.join(eps_dir, "instruction.txt"), 'w') as f:
            f.write(f"What action should the robot take to {lang}")
        
        proprio_ep_processed = proprio_ep.squeeze(axis=1)
        proprio_ep_processed = np.concatenate([proprio_ep_processed[:, :6], action_ep[:, 6:7]], 1)
        zarr.save(os.path.join(eps_dir, "proprio.zarr"), proprio_ep_processed)
        zarr.save(os.path.join(eps_dir, "instruction.zarr"), instruction_embedding)
        zarr.save(os.path.join(eps_dir, "action.zarr"), action_ep)

        for fid in range(0, len(action_ep)):
            total_frames += 1
            pil_img = Image.fromarray(img_ep[fid])
            pil_img_wrist = Image.fromarray(img_wrist_ep[fid])

            image_path = f"ep{ep_idx}-fid{fid}.jpg"
            image_path_wrist = f"ep{ep_idx}-fid{fid}_wrist.jpg"
            image_depth_path = f"ep{ep_idx}-fid{fid}_depth.png"
            # pil_img.save(f"{dataset_dir}/{image_path}")
            # pil_img_wrist.save(f"{dataset_dir}/{image_path_wrist}")
            # cv2.imwrite(f"{dataset_dir}/{image_depth_path}", img_ep[fid])

            if len(image_path_hist) == 0:
                for _ in range(history_len):
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

            exterior_images_dir = os.path.join(eps_dir, "images", "exterior_image_1_left")
            wrist_images_dir = os.path.join(eps_dir, "images", "wrist_image_left")
            os.makedirs(exterior_images_dir, exist_ok=True)
            os.makedirs(wrist_images_dir, exist_ok=True)

            fid_str = f"{fid:05d}"
            image_name = f"{fid_str}.jpg"
            pil_img.save(os.path.join(exterior_images_dir, image_name))
            pil_img_wrist.save(os.path.join(wrist_images_dir, image_name))

        ep_idx += 1

    print(f"Finished processing {total_frames} batches. Data saved to {dataset_dir}")