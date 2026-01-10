import json
from collections import Counter

# path = "/SSD_DISK/users/zhangjiahui/openvla/data/libero_10_no_noops_3d_97k_v2_mb_exp2_ws20_bs5.jsonl"
path = "/SSD_DISK/users/zhangjiahui/openvla/data/libero_10_no_noops_3d_97k_v2_mb_exp2_ws40_bs5.jsonl"

data = []
with open(path, "r") as f:
    for line in f:
        data.append(json.loads(line))

print("Number of dialogs: ", len(data))

diff_counter = Counter()
max_vis = Counter()

total_count = 0
total_traj = 0
for d in data:
    d_frame = [int(img.split(".")[0].split("-")[1][3:]) for img in d['image']]
    d_diff = [d_frame[-1] - frame for frame in d_frame]
    diff_counter.update(d_diff)
    total_count += len(d_diff)
    total_traj += 1
    max_vis.update([max(d_diff)])

# 打印统计结果和占比
for diff, count in diff_counter.items():
    percentage = (count / total_count) * 100
    print(f"diff {diff}: {count} times ({percentage:.2f}%)")

for diff, coint in max_vis.items():
    percentage = (coint / total_traj) * 100
    print(f"max diff {diff}: {coint} times ({percentage:.2f}%)")