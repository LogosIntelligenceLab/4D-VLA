import h5py
import os
import argparse
from collections import deque
from PIL import Image
import numpy as np
import json
import shutil

def process_fn(args: argparse.Namespace):
    hdf5_folder = args.hdf5_folder
    hdf5_path_list = [os.path.join(hdf5_folder, f) for f in os.listdir(hdf5_folder) if f.endswith(".hdf5")]

    for hdf5_path in hdf5_path_list:
        # Load HDF5 dataset
        assert os.path.exists(hdf5_path), f"Cannot find raw data file {hdf5_path}."
        orig_data_file = h5py.File(hdf5_path, "r")
        orig_data = orig_data_file["data"]
        if os.path.exists(args.target_dir):
            intput = input(f"Target directory {args.target_dir} already exists. Delete? (y/n): ")
            if intput.lower() != "y":
                print("Exiting...")
                return
            # delete the directory
            shutil.rmtree(args.target_dir)

        os.makedirs(args.target_dir, exist_ok=True)

        # Process dataset
        total_frames = 0
        ep_idx = 0
        json_data = []

        for i in range(len(orig_data.keys())):
            # Get data
            demo_data = orig_data[str(i)]   # TODO: Replace with actual data
            orig_actions = demo_data["actions"][()]
            orig_obs = demo_data["obs"][()]  # TODO: Replace with actual data

            action_hist = deque(maxlen=args.history_len)
            image_path_hist = deque(maxlen=args.history_len)
            is_zero_saved = False

            # Process data
            for fid in range(0, len(orig_actions)):
                total_frames += 1

                # Save images
                pil_img = Image.fromarray(orig_obs[fid])
                # TODO: Save wrist image
                # pil_img_wrist = Image.fromarray(img_wrist_ep[fid])

                pil_zero = Image.fromarray(np.zeros_like(orig_obs[fid]))

                image_path = f"ep{ep_idx}-fid{fid}.jpg"
                # TODO: Save wrist image
                # image_path_wrist = f"ep{ep_idx}-fid{fid}_wrist.jpg"

                pil_img.save(f"{args.target_dir}/{image_path}")
                # TODO: Save wrist image
                # pil_img_wrist.save(f"{args.target_dir}/{image_path_wrist}")

                if not is_zero_saved:
                    pil_zero = Image.fromarray(np.zeros_like(orig_obs[fid]))
                    image_path_zero = "zero.jpg"
                    pil_zero.save(f"{args.target_dir}/{image_path_zero}")
                    is_zero_saved = True

                # Save action
                formatted_pos = [f"{round((x+1.)*5000)}" for x in orig_actions[fid][:-1]]
                formatted_grip = [f"{round(orig_actions[fid][-1]+1.)}"]
                formatted_action = formatted_pos + formatted_grip

                # Save image history
                if len(image_path_hist) == 0:
                    for _ in range(args.history_len):
                        image_path_hist.append(image_path_zero)
                    image_path_hist.append(image_path)
                else:
                    image_path_hist.append(image_path)

                # Save entry
                entry = {
                    "id": total_frames,
                    "image": [],
                    "width_list": [],
                    "height_list": [],
                    "conversations": []
                }

                for i in range(args.history_len):
                    if (i+1) % args.sample_interval == 0:
                        entry['image'].append(image_path_hist[i])
                        entry['width_list'].append(args.image_width)
                        entry['height_list'].append(args.image_height)
                
                q = {
                    "from": "human",
                    # "value": "History:\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\nCurrent:\n<image>\n " + f"What action should the robot take to {lang}"
                    # "value": "History:\n<image>\n<image>\n<image>\n<image>\nCurrent:\n<image>\n " + f"What action should the robot take to {lang}"
                    "value": "History:\n:" + "<image>\n" * (args.history_len//args.sample_interval-1) +  "Current:\n<image>\n " + f"What action should the robot take to {lang}"
                }
                a = {
                    "from": "gpt",
                    "value": f"<action>[{', '.join(formatted_action)}]</action>"
                }
                entry["conversations"].append(q)
                entry["conversations"].append(a)

                json_data.append(entry)

            ep_idx += 1
    
    # ...

    # Save processed dataset
    len_data = len(json_data)
    jsonl_file_name = f'{args.dataset_name}_{len_data}len_v2_{args.history_len}history-v2-{args.sample_interval}sample_interval.jsonl'
    jsonl_file_path = os.path.join(args.target_dir, jsonl_file_name)

    with open(jsonl_file_path, "w") as jsonl_file:
        for entry in json_data:
            json.dump(entry, jsonl_file)
            jsonl_file.write('\n') 

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_folder", type=str, default="debug_for_demo/",
                        help="Path to directory containing raw HDF5 dataset.",)
    parser.add_argument("--target_dir", type=str, default="processed_data/",
                        help="Path to directory to save processed dataset.",)
    parser.add_argument("--dataset_name", type=str, default="debug_for_demo",
                        help="Name of dataset.",)
    parser.add_argument("--history_len", type=int, default=10,
                        help="Number of frames to include in history.", 
                        required=True)
    parser.add_argument("--sample_interval", type=int, default=1,
                        help="Interval at which to sample frames from history.",)
    parser.add_argument("--image_width", type=int, default=448,
                        help="Width of image.",)
    parser.add_argument("--image_height", type=int, default=448,
                        help="Height of image.",)
    
    args = parser.parse_args()

    # Process HDF5 dataset
    process_fn(args)