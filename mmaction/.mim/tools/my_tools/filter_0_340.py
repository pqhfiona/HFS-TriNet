import random

# 文件路径
file_path = '/data/pqh/dataset/US_1343/train_file_list.txt'

# 读取文件内容
with open(file_path, 'r') as file:
    lines = file.readlines()

# 分离标签为1和标签为0的数据
label_1_lines = [line for line in lines if line.strip().endswith('1')]
label_0_lines = [line for line in lines if line.strip().endswith('0')]

# 随机挑选340个标签为0的数据
random_label_0_lines = random.sample(label_0_lines, 340)

# 合并标签为1和随机挑选的标签为0的数据
selected_lines = label_1_lines + random_label_0_lines

# 打乱顺序以确保随机性
random.shuffle(selected_lines)

# 写入新的文件
output_file_path = '/data/pqh/dataset/US_1343/selected_train_file_list.txt'
with open(output_file_path, 'w') as output_file:
    for line in selected_lines:
        output_file.write(line)

print(f"已生成包含340个标签为1和340个标签为0的文件：{output_file_path}")