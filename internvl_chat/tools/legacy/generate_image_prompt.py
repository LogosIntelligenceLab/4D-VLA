import os
import json
import shutil


def generate_image_prompt(data_folder = '../data'):

    image_prompt_folder = os.path.join(data_folder, 'image_prompt')

    if not os.path.exists(image_prompt_folder):
        os.makedirs(image_prompt_folder)

    first_lines_dict = {}

    for filename in os.listdir(data_folder):
        if filename.endswith('incontext.jsonl'):
            file_path = os.path.join(data_folder, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
                try:
                    first_line_data = json.loads(first_line)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")
                    continue

                if 'image' in first_line_data and isinstance(first_line_data['image'], list) and first_line_data['image']:
                    image_path = os.path.join(data_folder, first_line_data['image'][0])
                    
                    if os.path.exists(image_path):
                        new_image_name = os.path.splitext(filename)[0] + '_image.jpg'
                        new_image_path = os.path.join(image_prompt_folder, new_image_name)
                        
                        shutil.copy(image_path, new_image_path)
                        print(f"Copied {image_path} to {new_image_path}")
                    else:
                        print(f"Image file not found: {image_path}")

                file_key = os.path.splitext(filename)[0]
                first_lines_dict[file_key] = {}
                first_lines_dict[file_key]['conversation'] = []
                first_lines_dict[file_key]['conversation'].append(first_line_data['conversations'][0])
                first_lines_dict[file_key]['conversation'].append(first_line_data['conversations'][1])


    jsonl_file_path = '../data/image_prompt.jsonl'

    with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
        for key, value in first_lines_dict.items():
            json_line = json.dumps({key: value}, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')

def generate_image_prompt_v2(data_folder = '../data', version=2):

    image_prompt_folder = os.path.join(data_folder, f'image_prompt_v{version}')

    if not os.path.exists(image_prompt_folder):
        os.makedirs(image_prompt_folder)

    first_lines_dict = {}

    for filename in os.listdir(data_folder):
        if filename.endswith('incontext_v2.jsonl'):
            file_path = os.path.join(data_folder, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for i in range(version):
                    f.readline()
                first_line = f.readline().strip()
                
                try:
                    first_line_data = json.loads(first_line)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")
                    continue

                if 'image' in first_line_data and isinstance(first_line_data['image'], list) and first_line_data['image']:
                    image_path = os.path.join(data_folder, first_line_data['image'][0])
                    
                    if os.path.exists(image_path):
                        new_image_name = os.path.splitext(filename)[0] + '_image.jpg'
                        new_image_path = os.path.join(image_prompt_folder, new_image_name)
                        
                        shutil.copy(image_path, new_image_path)
                        print(f"Copied {image_path} to {new_image_path}")
                    else:
                        print(f"Image file not found: {image_path}")

                file_key = os.path.splitext(filename)[0]
                first_lines_dict[file_key] = {}
                first_lines_dict[file_key]['conversation'] = []
                first_lines_dict[file_key]['conversation'].append('.'.join(first_line_data['conversations'][0]['value'].split('.')[:-1]))
                # first_lines_dict[file_key]['conversation'].append(first_line_data['conversations'][1])


    jsonl_file_path = f'../data/image_prompt_v{version}.jsonl'

    with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
        for key, value in first_lines_dict.items():
            json_line = json.dumps({key: value}, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')

if __name__ == "__main__":
    # generate_image_prompt()
    generate_image_prompt_v2(version=3)