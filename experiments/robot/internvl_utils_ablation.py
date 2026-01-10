"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
# import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from numpy import ndarray


# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Q99_LIBERO = [0.938, 0.876, 0.932, 0.104, 0.177, 0.146, 1.000]
Q01_LIBERO = [-0.745, -0.662, -0.938, -0.107, -0.207, -0.184, 0.000]
Q99_10 = [0.7714285850524902, 0.8464285731315613, 0.9375, 0.13928571343421936, 0.15964286029338837, 0.3246428668498993, 1.0]
Q01_10 = [-0.6348214149475098, -0.7741071581840515, -0.7633928656578064, -0.09749999642372131, -0.14819999992847435, -0.2742857038974762, 0.0]

Q99_10_3d = [0.7714285850524902, 0.8464285731315613, 0.9375, 0.13928571343421936, 0.15964286029338837, 0.33000001311302185, 1.0]
Q01_10_3d = [-0.6294642686843872, -0.7714285850524902, -0.7607142925262451, -0.09714286029338837, -0.14892856776714325, -0.2742857038974762, 0.0]

Q99_Spatial_3d = [0.9375, 0.8758928775787354, 0.9321428537368774, 0.1039285734295845, 0.17678570747375488, 0.14571428298950195, 1.0]
Q01_Spatial_3d = [-0.75, -0.6589285731315613, -0.9375, -0.10607142746448517, -0.20678570866584778, -0.18535713851451874, 0.0]

Q01_Goal_3d = [-0.8785714507102966, -0.7580357193946838, -0.9375, -0.1510714292526245, -0.16285714507102966, -0.13500000536441803, 0.0]
Q99_Goal_3d = [0.9375, 0.9133928418159485, 0.9375, 0.19821429252624512, 0.2626285624504117, 0.375, 1.0]

Q01_Object_3d = [-0.5383928418159485, -0.8758928775787354, -0.9375, -0.06964285671710968, -0.11678571254014969, -0.15964286029338837, 0.0]
Q99_Object_3d = [0.8464285731315613, 0.84375, 0.9375, 0.08142857253551483, 0.14892856776714325, 0.08571428805589676, 1.0]

MASK = [True, True, True, True, True, True, False]
GENERATE_CONFIG = dict(max_new_tokens=512, do_sample=False)


Q01_Spatial_Pose0 = [-0.8812500238418579, -0.3508928716182709, -0.9375, -0.1007142886519432, -0.2014285773038864, -0.207857146859169, 0.0]
Q99_Spatial_Pose0 = [0.9375, 0.6482142806053162, 0.918749988079071, 0.13500000536441803, 0.1842857152223587, 0.12535713613033295, 1.0]

Q01_Spatial_Pose_Large = [-0.8812500238418579, -0.3508928716182709, -0.9375, -0.1007142886519432, -0.2014285773038864, -0.207857146859169, 0.0]
Q99_Spatial_Pose_Large = [0.9375, 0.6482142806053162, 0.918749988079071, 0.13500000536441803, 0.1842857152223587, 0.12535713613033295, 1.0]

Q01_Spatial_Pose_Large_Uniform = [-0.8812500238418579, -0.3508928716182709, -0.9375, -0.1007142886519432, -0.2014285773038864, -0.207857146859169, 0.0]
Q99_Spatial_Pose_Large_Uniform = [0.9375, 0.6482142806053162, 0.918749988079071, 0.13500000536441803, 0.1842857152223587, 0.12535713613033295, 1.0]

Q01_MVBENCH = [-0.7473204946517944, -0.6895730948448181, -0.9375, -0.1093504148721695, -0.19964531183242798, -0.1854198694229126, 0.0]
Q99_MVBENCH = [0.9584454560279843, 0.872106018066406, 0.9294642806053162, 0.10605272293090816, 0.17427105784416191, 0.11888263791799532, 1.0]

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12, convert_to_rgb=True):
    if convert_to_rgb:
        image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_vla_internvl(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    if cfg.load_in_8bit:
        model = AutoModel.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            device_map='cuda',
            trust_remote_code=True
        ).eval()
        print("[*] Loaded in 8-bit mode")
    elif cfg.load_in_4bit:
        model = AutoModel.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            device_map='cuda',
            trust_remote_code=True
        ).eval()
        print("[*] Loaded in 4-bit mode")
    else:
        model = AutoModel.from_pretrained(
                cfg.pretrained_checkpoint,
                torch_dtype=torch.bfloat16,
                # low_cpu_mem_usage=True,
                # use_flash_attn=True,
                attn_implementation="flash_attention_2",
                # device_map='cuda',
                trust_remote_code=True
            ).eval().cuda()
        print("[*] Loaded in 16-bit mode")
    for param in model.parameters():
        param.requires_grad = False

    return model



def get_vla_action_internvl_ablation(model, 
                            processor, 
                            obs, 
                            task_label, 
                            center_crop=False, 
                            use_image_prompt=False, 
                            prompt_version='v1', 
                            use_2view=False, 
                            use_anyview=False,
                            use_nromaction=False,
                            use_hist=False,
                            hist_version='v1',
                            hist_len=0,
                            use_multi_size=False,
                            all_224=False,
                            sample_interval=1,
                            use_action_hist=False,
                            use_depth=False,
                            use_proprio=False,
                            is_ft=False,
                            t=0,
                            task_suite_name=None):
    """Generates an action with the VLA policy."""
    image = obs["full_image"]
    image: Image.Image = Image.fromarray(image)
    image = image.convert("RGB")


    num_patches_list = None
    history = None
    input_size = 224 if all_224 else 448
    pixel_values = load_image(image, max_num=1, convert_to_rgb=False, input_size=input_size).to(torch.bfloat16).cuda()
    
    prompt = task_label
    if obs.get('image_primary_hist', None) is None or obs['image_primary_hist'].shape[0] == 1:
        use_hist = False
    else:
        use_hist = True
        hist_len = obs['image_primary_hist'].shape[0]
    
    if use_hist:
        image_primary_hist = obs['image_primary_hist']  # hist, current
        image_primary_hist_list = [Image.fromarray(image_primary_hist[im]).convert("RGB") for im in range(hist_len-1)]
        hist_input_size = 224 if use_multi_size else 448
        pixel_values_hist = [load_image(im, max_num=1, convert_to_rgb=False, input_size=hist_input_size).to(torch.bfloat16).cuda() for im in image_primary_hist_list]
        

        if use_2view:
            image_wrist = obs['wrist_image']
            image_wrist: Image.Image = Image.fromarray(image_wrist)
            image_wrist = image_wrist.convert("RGB")

            pixel_values_wrist = load_image(image_wrist, max_num=6, convert_to_rgb=False).to(torch.bfloat16).cuda()

            num_patches_list = [im.size(0) for im in pixel_values_hist] + [pixel_values.size(0)] + [pixel_values_wrist.size(0)]
            pixel_values = torch.cat(pixel_values_hist + [pixel_values] + [pixel_values_wrist], dim=0)
            question = "History[One View]:\n<image>\n<image>\n<image>\n<image>\nCurrent[Two Views]:\n<image>\n<image>\n " + f"What action should the robot take to {prompt}"
        else:
            if not use_multi_size:
                num_patches_list = [im.size(0) for im in pixel_values_hist] + [pixel_values.size(0)]
                pixel_values = torch.cat(pixel_values_hist + [pixel_values], dim=0)
            else:
                if sample_interval != 1:
                    pixel_values_hist_ = []
                    for index in range(len(pixel_values_hist)):
                        if (index + 1) % sample_interval == 0:
                            pixel_values_hist_.append(pixel_values_hist[index])
                    pixel_values_hist = pixel_values_hist_
                num_patches_list = [pixel_values.size(0)]
                num_patches_list_hist = [im.size(0) for im in pixel_values_hist]
                pixel_values_hist = torch.cat(pixel_values_hist, dim=0) if pixel_values_hist else None
            
            # question = "History:\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\nCurrent:\n<image>\n " + f"What action should the robot take to {prompt}"
            # q_image = ""
            # for _ in range(hist_len// sample_interval - 1):
            #     q_image = q_image + "<image>\n"
            q_image = "<image>\n" * (hist_len // sample_interval - 1)
            question = "History:\n" + q_image + "Current:\n<image>\n " + f"What action should the robot take to {prompt}"
    else:
        question = f"History:\nCurrent:\n<image>\n What action should the robot take to {prompt}"

        # else:
        #     raise NotImplementedError


    if use_depth and (not use_proprio) and (not use_hist):
        response = model.chat(processor, 
                                pixel_values, 
                                question, 
                                GENERATE_CONFIG, 
                                history=history, 
                                num_patches_list=num_patches_list,
                                # num_patches_list_hist=num_patches_list_hist if use_multi_size else None,
                                # pixel_values_hist=pixel_values_hist,
                                depth=obs['depth'],
                                # depth_hist=obs['depth_hist'],
                                depth_mask=obs['depth_mask'],
                                # depth_mask_hist=obs['depth_mask_hist'],
                                pose=obs['pose'],
                                intrinsics=obs['intrinsics'],
                                # frame_id=obs['frame_id'],
                                )
    
    elif use_depth and (not use_proprio) and use_hist:
        response = model.chat(processor, 
                                pixel_values, 
                                question, 
                                GENERATE_CONFIG, 
                                history=history, 
                                num_patches_list=num_patches_list,
                                num_patches_list_hist=num_patches_list_hist if use_multi_size else None,
                                pixel_values_hist=pixel_values_hist,
                                depth=obs['depth'],
                                depth_hist=obs['depth_hist'],
                                depth_mask=obs['depth_mask'],
                                depth_mask_hist=obs['depth_mask_hist'],
                                pose=obs['pose'],
                                intrinsics=obs['intrinsics'],
                                frame_id=obs['frame_id'],)
    elif use_proprio and (not use_depth):
        # proprio = obs['proprio_hist']# [:-1]
        # if is_ft:
        #     proprio = obs['proprio_hist'][9:10, 0:6]
        # else:
        if obs.get('proprio_hist', None) is None:
            proprio = obs['proprio'][:6]
        else:
            proprio = obs['proprio_hist'][5:10, 0:6]
            
        response = model.chat(processor, 
                            pixel_values, 
                            question, 
                            GENERATE_CONFIG, 
                            history=history, 
                            num_patches_list=num_patches_list,
                            # num_patches_list_hist=num_patches_list_hist if use_multi_size else None,
                            # pixel_values_hist=pixel_val/ues_hist,
                            # depth=obs['depth'],
                            # depth_hist=obs['depth_hist'],
                            # depth_mask=obs['depth_mask'],
                            # depth_mask_hist=obs['depth_mask_hist'],
                            # pose=obs['pose'],
                            # intrinsics=obs['intrinsics'],
                            proprio=proprio)

    elif use_proprio and use_depth:
        # proprio = obs['proprio_hist']# [:-1]
        # if is_ft:
        #     proprio = obs['proprio_hist'][9:10, 0:6]
        # else:
        if obs.get('proprio_hist', None) is None:
            proprio = obs['proprio'][:6]
        else:
            proprio = obs['proprio_hist'][5:10, 0:6]

        response = model.chat(processor, 
                            pixel_values, 
                            question, 
                            GENERATE_CONFIG, 
                            history=history, 
                            num_patches_list=num_patches_list,
                            # num_patches_list_hist=num_patches_list_hist if use_multi_size else None,
                            # pixel_values_hist=pixel_values_hist,
                            depth=obs['depth'],
                            # depth_hist=obs['depth_hist'],
                            # depth_mask=obs['depth_mask'],
                            # depth_mask_hist=obs['depth_mask_hist'],
                            pose=obs['pose'],
                            intrinsics=obs['intrinsics'],
                            proprio=proprio)
    elif use_hist and use_hist:
        if obs.get('frame_id', None) is not None:
            response = model.chat(processor, 
                                    pixel_values, 
                                    question, 
                                    GENERATE_CONFIG, 
                                    history=history, 
                                    num_patches_list=num_patches_list,
                                    num_patches_list_hist=num_patches_list_hist if use_multi_size else None,
                                    pixel_values_hist=pixel_values_hist,
                                    frame_id=obs['frame_id'],
                                    )
        else:
            response = model.chat(processor, 
                                    pixel_values, 
                                    question, 
                                    GENERATE_CONFIG, 
                                    history=history, 
                                    num_patches_list=num_patches_list,
                                    num_patches_list_hist=num_patches_list_hist if use_multi_size else None,
                                    pixel_values_hist=pixel_values_hist,
                                    )
    else:
        response = model.chat(processor, 
                                pixel_values, 
                                question, 
                                GENERATE_CONFIG, 
                                history=history, 
                                num_patches_list=num_patches_list,
                            #   num_patches_list_hist=num_patches_list_hist if use_multi_size else None,
                            #   pixel_values_hist=pixel_values_hist,
                                )
    
    
    
    # print("response: ", response)
    if isinstance(response, str):
        normalized_actions = json.loads(response)
        normalized_actions = np.array(normalized_actions).reshape(1, 7)

        # if use_nromaction:
        pos = (normalized_actions[:, :-1] / 5000) - 1
        grisp = normalized_actions[:, -1] - 1
        normalized_actions = np.concatenate((pos, [grisp]), axis=1)
    elif isinstance(response, ndarray):
        # print("response: ", response)   
        response[:, -1] = 1.0 * (response[:, -1] > 0.5)
        normalized_actions = response
        


    if task_suite_name == "libero_spatial":
        Q99, Q01 = Q99_Spatial_3d, Q01_Spatial_3d
    elif task_suite_name == "libero_object":
        Q99, Q01 = Q99_Object_3d, Q01_Object_3d
    elif task_suite_name == "libero_goal":
        Q99, Q01 = Q99_Goal_3d, Q01_Goal_3d
    elif task_suite_name == "libero_10":
        Q99, Q01 = Q99_10_3d, Q01_10_3d
    
    # if is_ft:
    #     action = normalized_actions[0]
    #     action[0:3] = action[0:3] / 2
    #     action[3:6] = action[3:6] / 5
    # else:
    action_high, action_low = np.array(Q99), np.array(Q01)
    action = np.where(
        MASK,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    action = action[0]

    # action = normalized_actions[0]
    # action[3:6] = np.zeros_like(action[3:6])
    # action[0:3] = action[0:3] * 3
    # print("action: ", action)
    return action, response
