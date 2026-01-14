import os
import random

# 设置随机种子以保证可重复性
random.seed(42)

# 读取数据文件
data_file = "/data/pqh/dataset/US_1343/all_1343.txt"
with open(data_file, "r") as f:
    lines = f.readlines()

# 按类别分组
class_0 = []  # 类别 0 的数据
class_1 = []  # 类别 1 的数据

for line in lines:
    filename, num, label = line.strip().split()
    if int(label) == 0:
        class_0.append(line)
    else:
        class_1.append(line)

# 按 7:3 的比例划分数据
def split_data(data, train_ratio=0.7):
    random.shuffle(data)  # 打乱数据
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data

# 划分类别 0 和类别 1 的数据
train_0, test_0 = split_data(class_0)
train_1, test_1 = split_data(class_1)

# 合并训练集和测试集
train_data = train_0 + train_1
test_data = test_0 + test_1

# 打乱训练集和测试集（可选）
random.shuffle(train_data)
random.shuffle(test_data)

# 保存训练集和测试集
def save_data(data, filename):
    with open(filename, "w") as f:
        for line in data:
            f.write(line)

save_data(train_data, "train.txt")
save_data(test_data, "test.txt")

# 打印统计信息
print(f"Total samples: {len(lines)}")
print(f"Class 0: {len(class_0)} (Train: {len(train_0)}, Test: {len(test_0)})")
print(f"Class 1: {len(class_1)} (Train: {len(train_1)}, Test: {len(test_1)})")
print(f"Train set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")