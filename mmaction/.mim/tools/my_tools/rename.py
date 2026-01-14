import os

# 指定目录路径
folder_path = '/data/pqh/dataset/US_+340_1/US_+340_1'

# 遍历目录下的所有子文件夹
for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        # 新的文件夹名字（加上 "_1"）
        new_folder_name = subfolder + '_1'
        new_subfolder_path = os.path.join(folder_path, new_folder_name)

        # 重命名文件夹
        os.rename(subfolder_path, new_subfolder_path)