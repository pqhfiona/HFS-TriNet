"""
2025.3.5  查看有两个txt文件有没有重叠的行
"""
def compare_files(file1, file2):
    # 读取第一个文件的内容
    with open(file1, 'r') as f1:
        lines1 = set(f1.readlines())

    # 读取第二个文件的内容
    with open(file2, 'r') as f2:
        lines2 = set(f2.readlines())

    # 找出重叠的行
    overlapping_lines = lines1.intersection(lines2)

    # 打印重叠的行
    if overlapping_lines:
        print("以下行在两个文件中重叠：")
        for line in overlapping_lines:
            print(line.strip())
    else:
        print("两个文件没有重叠的行。")

# 文件路径
file1 = '/data/pqh/dataset/US_1343/train_file_list.txt'
file2 = '/data/pqh/dataset/US_1343/val_file_list.txt'

# 调用函数比较文件
compare_files(file1, file2)