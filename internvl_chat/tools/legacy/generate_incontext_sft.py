# setsid nohup python tools/process_rlds.py > process_rlds_bridge.log 2>&1 &

import os
import json
import random
from tqdm import tqdm


def load_jsonl_files_from_folder(folder_path):
    json_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl') and ("incontext" not in filename) and ("prompt" not in filename):
            dataset_name = filename.split('.')[0]
            file_path = os.path.join(folder_path, filename)
            print(f"Loading: {file_path}")
            dataset_list = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    dataset_list.append(json.loads(line))
            json_data[dataset_name] = dataset_list

    return json_data

def create_incontext_conversation_v1(data, dataset_name, index):
    example = random.choice(data)
    qa = random.choice(data)
    while qa == example:
        qa = random.choice(data)

    example_path = f'{dataset_name}/images_v2/{example["image"]}'
    qa_path = f'{dataset_name}/images_v2/{qa["image"]}'


    incontext_conversation = {
        "id": index,
        "image": [example_path, qa_path],
        "width_list": [example['width'], qa['width']],
        "height_list": [example['height'], qa['height']],
        "conversations": []
    }
    example_instructions = "<image>\nTask: You are controlling a robot. Example: " + example['conversations'][0]['value'].split("\n")
    example_action = example['conversations'][1]['value']

    qa_instructions = "<image>\nTask: Now, please determine the next action for the robot. " + qa['conversations'][0]['value'].split("\n")[1]
    qa_action = qa['conversations'][1]['value']


    
    incontext_conversation["conversations"].append({"from": "human", "value": example_instructions})
    incontext_conversation["conversations"].append({"from": "gpt", "value": example_action})

    incontext_conversation["conversations"].append({"from": "human", "value": qa_instructions})
    incontext_conversation["conversations"].append({"from": "gpt", "value": qa_action})

    return incontext_conversation

def create_incontext_conversation_v2(data, dataset_name, index):
    example = random.choice(data)
    qa = random.choice(data)
    while qa == example:
        qa = random.choice(data)

    example_path = f'{dataset_name}/images_v2/{example["image"]}'
    qa_path = f'{dataset_name}/images_v2/{qa["image"]}'


    incontext_conversation = {
        "id": index,
        "image": [example_path, qa_path],
        "width_list": [example['width'], qa['width']],
        "height_list": [example['height'], qa['height']],
        "conversations": []
    }
    example_instructions = "Task: You are controlling a robot. Example: " + example['conversations'][0]['value'].split("\n")[1] + " in \n<image>"
    example_action = example['conversations'][1]['value']

    example = example_instructions + "\nOutput: " + example_action

    qa_instructions = example + "\n" + "Task: Now, please determine the next action for the robot. " + qa['conversations'][0]['value'].split("\n")[1] + " in \n<image>"
    qa_action = qa['conversations'][1]['value']


    
    # incontext_conversation["conversations"].append({"from": "human", "value": example_instructions})
    # incontext_conversation["conversations"].append({"from": "gpt", "value": example_action})

    incontext_conversation["conversations"].append({"from": "human", "value": qa_instructions})
    incontext_conversation["conversations"].append({"from": "gpt", "value": qa_action})

    return incontext_conversation

if __name__ == "__main__":
    pass
    num_data = {}
    folder_path = '../data'
    all_data = load_jsonl_files_from_folder(folder_path)

    total_data_to_generate = 1_000_000

    num_repeat_per_dataset = {
        'libero_10_no_noops_96k_v2': 4,
        'libero_goal_no_noops_49k_v2': 4,
        'libero_object_no_noops_63k_v2': 4,
        'libero_spatial_no_noops_50k_v2': 4,
        'bridge_orig_1305k_v2': 1,
        'calvin_rlds_1289k_v2': 1
    }

    num_data_per_dataset = {}
    for dataset_name in all_data.keys():
        num_data_per_dataset[dataset_name] = len(all_data[dataset_name]) * num_repeat_per_dataset[dataset_name]
    total_data = sum(num_data_per_dataset.values())
    print(f"Total data: {total_data}")

    num_data_per_dataset_to_generate = {}
    for dataset_name in all_data.keys():
        weight = num_data_per_dataset[dataset_name] / total_data
        num_data_per_dataset_to_generate[dataset_name] = int(total_data_to_generate * weight)
        print(f"Dataset: {dataset_name}, Weight: {weight}, Num data to generate: {num_data_per_dataset_to_generate[dataset_name]}")

    print(f"Total data to generate: {sum(num_data_per_dataset_to_generate.values())}")
    

    # for dataset_name in all_data.keys():
    for dataset_name in num_data_per_dataset_to_generate.keys():
        print(f"Dataset: {dataset_name}")
        data = all_data[dataset_name]

        incontext_list = []
        for i in tqdm(range(num_data_per_dataset_to_generate[dataset_name])):
            incontext_conversation = create_incontext_conversation_v2(data, 
                                                                   dataset_name="_".join(dataset_name.split("_")[:-2]), 
                                                                   index=i)
            incontext_list.append(incontext_conversation)

        jsonl_file_name = f'{dataset_name}_incontext_v2.jsonl'
        jsonl_file_path = os.path.join(folder_path, jsonl_file_name)

        with open(jsonl_file_path, "w") as json_file:
            for entry in incontext_list:
                json.dump(entry, json_file)
                json_file.write('\n') 

        print(f"Finished processing {dataset_name}. Saved to {jsonl_file_path}")
        # break










    
        

    
