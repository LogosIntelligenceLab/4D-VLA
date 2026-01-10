import json
import os

def round_floats(obj, decimal_places=4):
    if isinstance(obj, list):
        return [round_floats(i, decimal_places) for i in obj]
    elif isinstance(obj, float):
        return round(obj, decimal_places)
    return obj  

if __name__ == "__main__":
    raw_folder_path = "/SSD_DISK/users/zhangjiahui/openvla/data/droid/processed_jsonl_folder_20ws_10bs"
    output_folder_path = "/SSD_DISK/users/zhangjiahui/openvla/data/droid/processed_jsonl_folder_20ws_5bs_simple"

    for folder in os.listdir(raw_folder_path):
        folder_path = os.path.join(raw_folder_path, folder)
        output_folder_path_ = os.path.join(output_folder_path, folder)
        if not os.path.exists(output_folder_path_):
            os.makedirs(output_folder_path_)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            output_file_path_ = os.path.join(output_folder_path_, file)

            new_data_list = []
            with open(file_path, "r", encoding="utf-8") as file:
                data_list = [json.loads(line) for line in file]
            
            for data in data_list:
                path_prefix = os.path.join(*data['image'][0].split("/")[:-1])
                data['path_prefix'] = path_prefix
                data['image'] = [d.split("/")[-1] for d in data['image']]
                data['image_depth'] = [d.split("/")[-1] for d in data['image_depth']]
                data['action'] = [round(a, 4) for a in data['action']]
                # data['proprio'] = [round(p, 4) for p in data['proprio']]
                del data['proprio']
                data['camera_info'] = {key: round_floats(value) for key, value in data['camera_info'].items()}
                new_data_list.append(data)

            with open(output_file_path_, "w", encoding="utf-8") as file:
                for item in new_data_list:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Finish processing {file_path} and save to {output_file_path_}")
    print(" Finish all processing")