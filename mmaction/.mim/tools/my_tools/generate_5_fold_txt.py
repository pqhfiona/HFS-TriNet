import os
import os.path as osp
import numpy as np
from sklearn.model_selection import KFold

# 数据集路径
ann_file_train = '/data/pqh/dataset/US_1343/all.txt'  # 原始训练集文件
output_dir = '/data/pqh/dataset/US_1343/kfold_splits'  # 保存划分结果的目录

# 加载训练数据的文件名列表
with open(ann_file_train, 'r') as f:
    file_list = f.readlines()

file_list = np.array(file_list)

# 初始化五折交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 进行五折交叉验证划分
for fold, (train_idx, val_idx) in enumerate(kfold.split(file_list)):
    print(f"Fold {fold + 1}")

    # 划分训练集和验证集
    train_files = file_list[train_idx]
    val_files = file_list[val_idx]

    # 保存当前折的训练集和验证集到文件
    train_ann_file = osp.join(output_dir, f'train_fold_{fold + 1}.txt')
    val_ann_file = osp.join(output_dir, f'val_fold_{fold + 1}.txt')

    with open(train_ann_file, 'w') as f:
        f.writelines(train_files)

    with open(val_ann_file, 'w') as f:
        f.writelines(val_files)

    print(f"Fold {fold + 1} 的训练集已保存到: {train_ann_file}")
    print(f"Fold {fold + 1} 的验证集已保存到: {val_ann_file}")