import pandas as pd
import matplotlib.pyplot as plt


# 创建数据字典
data = {
    'epoch': [1, 2, 3, 4, 5],
    'ft-spatialonly': [
        [0.75, 0.85, 0.7, 0.9, 0.65, 0.77],
        [0.65, 0.9, 0.75, 0.8, 0.85, 0.79],
        [0.9, 0.85, 0.7, 0.85, 0.65, 0.79],
        [0.9, 0.8, 0.7, 0.8, 0.85, 0.81],
        [0.8, 0.65, 0.85, 0.9, 0.55, 0.75]
    ],
    'sft-spatialonly': [
        [0.95, 0.75, 0.45, 0.95, 0.55, 0.73],
        [0.8, 0.8, 0.9, 0.85, 0.55, 0.78],
        [0.9, 0.65, 0.65, 0.75, 0.75, 0.74],
        [0.6, 0.8, 0.7, 0.75, 0.85, 0.74],
        [0.7, 0.65, 0.85, 0.65, 0.55, 0.68]
    ],
    'sft-mask-spatialonly': [
        [1.0, 0.85, 0.7, 0.65, 0.55, 0.75],
        [0.45, 0.7, 0.85, 0.85, 0.7, 0.71],
        [0.85, 0.75, 0.7, 0.85, 0.7, 0.77],
        [0.8, 0.8, 0.9, 0.85, 0.6, 0.79],
        [0.75, 0.75, 0.8, 0.8, 0.55, 0.73]
    ],
    'sft-cosine-spatialonly': [
        [0.9, 0.75, 0.6, 0.8, 0.7, 0.75],
        [0.55, 0.9, 0.8, 0.75, 0.4, 0.68],
        [0.8, 0.85, 0.7, 0.7, 0.6, 0.73],
        [0.8, 0.8, 0.65, 0.85, 0.6, 0.74],
        [0.65, 0.85, 0.85, 0.8, 0.55, 0.74]
    ],
    'new_data': [
        [0.85, 1.0, 0.75, 1.0, 0.75, 0.87],  # openvla-finetune-crop_true-official
        [0.75, 0.8, 0.85, 1.0, 0.7, 0.82],   # exp10-pretrain
        [0.6, 0.7, 0.75, 0.8, 0.55, 0.68],    # sftv1-10ksteps
        [0.8, 0.7, 0.6, 0.85, 0.4, 0.67],     # sftv1-20ksteps
        [0.85, 0.85, 0.7, 0.85, 0.6, 0.77]    # sftv2-10ksteps
    ]
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 画图
tasks = ['ft-spatialonly', 'sft-spatialonly', 'sft-mask-spatialonly', 'sft-cosine-spatialonly']
new_task_labels = ['openvla-finetune-crop_true-official', 'exp10-pretrain', 'sftv1-10ksteps', 'sftv1-20ksteps', 'sftv2-10ksteps']
for i in range(6):
    plt.figure()
    for task in tasks:
        plt.plot(df['epoch'], [df[task][j][i] for j in range(len(df))], marker='o', label=task)
    plt.scatter([1, 2, 3, 4, 5], df['new_data'][i], marker='x', color='red', label=new_task_labels[i])
    plt.title(f'Task {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.xticks(df['epoch'])
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(f'{i}_performance.png')
