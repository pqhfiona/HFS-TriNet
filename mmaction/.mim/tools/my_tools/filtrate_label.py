"""
2024.5.14  过滤txt文件中标签一样的，0或者1
"""



# 输入文件路径和输出文件路径
input_file = '/data/pqh/dataset/US_1343/train_file_list.txt'
output_file = '/data/pqh/dataset/US_1343/train_file_list_1.txt'

# 打开输入文件以及输出文件
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # 遍历输入文件的每一行
    for line in infile:
        parts = line.strip().split()  # 分割每一行的内容
        # 检查第三列是否为1
        if len(parts) >= 3 and parts[2] == '1':
            # 将满足条件的行写入到输出文件
            outfile.write(line)
