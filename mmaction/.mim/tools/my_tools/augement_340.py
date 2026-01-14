import os
import cv2
import numpy as np
import random

# 路径设置
txt_file_path = "/data/pqh/dataset/US_1343/train_file_list_1.txt"  # 替换为你的txt文件路径
all_files_path = "/data/pqh/dataset/US_1343/all"  # 替换为包含所有文件的路径
output_path = "/data_nas/pqh/dataset/US_+340_1/US_+340_1"  # 替换为保存增强文件的路径


# 创建输出目录
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 读取txt文件
def read_txt_file(txt_file_path):
    file_list = []
    with open(txt_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                file_name, frame_count, label = parts
                file_list.append((file_name, int(frame_count), int(label)))
    return file_list

# 数据增强函数
def data_augmentation(frame):
    # 随机选择一种增强方法
    method = random.choice(["flip", "rotate", "contrast"])

    if method == "flip":
        # 翻转
        return cv2.flip(frame, 1)  # 水平翻转
    elif method == "rotate":
        # 旋转
        rows, cols, _ = frame.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)  # 旋转45度
        return cv2.warpAffine(frame, M, (cols, rows))
    elif method == "contrast":
        # 增加对比度
        return cv2.convertScaleAbs(frame, alpha=1.5, beta=0)  # alpha控制对比度

# 处理每个视频帧序列
def process_video_frames(file_name, frame_count, all_files_path, output_path):
    # 构建视频帧序列的目录路径
    video_frame_dir = os.path.join(all_files_path, file_name)
    if not os.path.exists(video_frame_dir):
        print(f"视频帧目录 {video_frame_dir} 不存在，跳过")
        return

    # 获取视频帧序列中的所有图片文件
    frame_files = sorted([f for f in os.listdir(video_frame_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))])
    if len(frame_files) == 0:
        print(f"视频帧目录 {video_frame_dir} 中没有图片文件，跳过")
        return

    # 创建保存增强帧的目录
    save_dir = os.path.join(output_path, file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 处理每一帧
    for frame_file in frame_files:
        frame_path = os.path.join(video_frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"无法读取文件 {frame_path}，跳过")
            continue

        # 数据增强
        enhanced_frame = data_augmentation(frame)

        # 保存增强后的帧（文件名不变）
        save_path = os.path.join(save_dir, frame_file)
        cv2.imwrite(save_path, enhanced_frame)

    print(f"视频帧序列 {file_name} 处理完成，共处理 {len(frame_files)} 帧")

# 主函数
if __name__ == "__main__":
    # 读取txt文件
    file_list = read_txt_file(txt_file_path)
    # 处理每个视频帧序列
    for file_name, frame_count, label in file_list:
        process_video_frames(file_name, frame_count, all_files_path, output_path)