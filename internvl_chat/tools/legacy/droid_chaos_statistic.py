import os
import json
import random
import numpy as np
import torch

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_snake_case(prompt: str) -> str:
    # 替换空格为下划线
    snake = prompt.replace(" ", "_")
    # 替换括号中的空格
    snake = snake.replace("(", "(_").replace(")", "_)")
    # 替换句号和逗号后跟空格的情况（避免断句）
    snake = snake.replace(",_", ",").replace("._", ".")
    return snake


if __name__ == "__main__":
    set_seed(42)

    droid_json_path = "/SSD_DISK/users/zhangjiahui/openvla/internvl_chat/shell/exp_data/droid_pretrain_mb_20ws_5bs_simple.json"
    with open(droid_json_path, "r") as f:
        droid_json = json.load(f)
    # print(droid_json)
    item_list = []

    for i, lab in enumerate(droid_json):
        # if i == 2:
        #     break
        data_item = droid_json[lab]
        root = data_item["root"]
        annotation = data_item["annotation"]
        prefix = "/SSD_DISK/users/zhangjiahui/openvla"
        annotation = annotation.replace("..", prefix)
        new_root = root.replace("..", prefix)
        with open(annotation, "r") as f:
            annotations = [json.loads(line) for line in f.readlines()]
        print()
        sampled_ann = random.sample(annotations, 50)
        del annotations
        
        for itm in sampled_ann:
            prompt = itm['conversations'][0]['value'][55:]
            item_list.append({
                "image": os.path.join(new_root, itm['path_prefix'], itm["image"][-1]),
                "prompt": prompt,
            })
    print("item_list", len(item_list))
    sampled_path = "sampeld_droid"
    os.makedirs(sampled_path, exist_ok=True)
    for i, itm in enumerate(item_list):
        prompt = itm["prompt"]
        prompt = to_snake_case(prompt)
        file_name = os.path.basename(itm["image"])
        # file_name = f"{i}_{file_name}_{prompt}.jpg"
        file_name = f"{i}.jpg"
        new_path = os.path.join(sampled_path, file_name)
        os.system(f"cp {itm['image']} {new_path}")
    data_jsonl_path = "sampeld_droid/data.jsonl"
    with open(data_jsonl_path, "w") as f:
        for item in item_list:
            json_line = json.dumps(item)
            f.write(json_line + "\n")


        # print(itm["image"], new_path)

    # data_folder = "/SSD_DISK/users/zhangjiahui/openvla/data/droid/droid_processed_data"
    # lab_folder = os.listdir(data_folder)
    # total_img_path = []
    # for lab in lab_folder:
    #     lab_img_path = []
    #     lab_path = os.path.join(data_folder, lab)
    #     if not os.path.isdir(lab_path):
    #         continue
    #     print(lab_path)
    #     lab_data = os.listdir(lab_path)
    #     for data in lab_data:
    #         data_path = os.path.join(lab_path, data)
    #         if not os.path.isdir(data_path):
    #             continue
    #         # print(data_path)
    #         sub_data_files = os.listdir(data_path)
    #         for sub_data_file in sub_data_files:
    #             sub_img_foler_path = os.path.join(data_path, sub_data_file, "images")
    #             meta_json = os.path.join(data_path, sub_data_file, "metadata.json")
    #             with open(meta_json, "r") as f:
    #                 meta_json = json.load(f)
    #             prompt = meta_json["current_task"]
    #             prompt = to_snake_case(prompt)

    #             if not os.path.isdir(sub_img_foler_path):
    #                 continue
    #             # img_path = [os.path.join(sub_img_foler_path, img) for img in os.listdir(sub_img_foler_path)]
    #             # lab_img_path.extend(img_path)
    #             for sub_sub_data_file in os.listdir(sub_img_foler_path):
    #                 sub_sub_data_file_path = os.path.join(sub_img_foler_path, sub_sub_data_file, "left")
    #                 img_path = [os.path.join(sub_sub_data_file_path, img) for img in os.listdir(sub_sub_data_file_path)]
    #                 try:
    #                     sampled = random.sample(img_path, 10)
    #                 except:
    #                     sampled = []
    #                     print(sub_sub_data_file_path, "not enough images")
    #                 # if not os.path.isfile(sub_sub_data_file_path):
    #                 #     continue
    #                 new_sampled = []
    #                 for itm in sampled:
    #                     itm = f"{itm}.{prompt}"
    #                     new_sampled.append(itm)
                    

    #                 lab_img_path.extend(new_sampled)

    #     try:
    #         sampled_img_path = random.sample(lab_img_path, 50)
    #     except:
    #         sampled_img_path = []
    #         print(lab_path, "not enough images")
    #     total_img_path.extend(sampled_img_path)
    # print("total_img_path", len(total_img_path))

    # sampled_path = "sampeld_droid"
    # os.makedirs(sampled_path, exist_ok=True)
    # for i, path in enumerate(total_img_path):
    #     file_name = os.path.basename(path)
    #     file_name = f"{i}_{file_name}.jpg"
    #     new_path = os.path.join(sampled_path, file_name)
    #     os.system(f"cp {path} {new_path}")

        