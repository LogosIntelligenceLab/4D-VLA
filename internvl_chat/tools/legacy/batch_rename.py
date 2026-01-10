import os

if __name__ == "__main__":
    dir_path = "/SSD_DISK/users/zhangjiahui/openvla/data/franka/replay_traj_sampled_task2_30-49"
    
    sub_folders = os.listdir(dir_path)
    for sub_folder in sub_folders:
        old_path = os.path.join(dir_path, sub_folder)
        if os.path.isdir(old_path) and sub_folder.isdigit():
            new_index = int(sub_folder) + 30
            new_name = f"{new_index:03d}"  # 保证格式为三位数
            new_path = os.path.join(dir_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed {sub_folder} -> {new_name}")