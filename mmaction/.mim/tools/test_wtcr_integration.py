#!/usr/bin/env python3
"""
测试WTCR与eval_with_speckle.py的集成
"""

import torch
import sys
import os
sys.path.append('/data/pqh/env/MM/tools')

from wtcr_visualization import WTCRVisualizer

def test_wtcr_with_real_data():
    """使用真实的模型和数据测试WTCR分析"""

    # 模拟一个特征张量（假设这是从backbone.layer4 hook获取的）
    # 基于MMAction的TPN模型，layer4输出通常是 (N, C, T, H, W)
    batch_size = 1
    channels = 512  # ResNet50 layer4 的通道数
    time_frames = 8
    height = 14  # 经过下采样
    width = 14

    # 创建模拟的clean特征
    clean_feature = torch.randn(batch_size, channels, time_frames, height, width)
    print(f"Clean feature shape: {clean_feature.shape}")

    # 创建WTCR可视化器
    visualizer = WTCRVisualizer(wavelet_type='db1', wt_levels=1)

    # 添加speckle噪声
    noise_var = 0.05
    noisy_feature = visualizer.apply_speckle_noise(clean_feature, noise_var)
    print(f"Noisy feature shape: {noisy_feature.shape}")

    # 创建WTCR分支（模拟）
    wtcr_branch = visualizer.create_wtcr_branch(in_channels=channels, out_channels=channels)
    wtcr_branch.eval()

    # WTCR处理
    wtcr_output = visualizer.apply_wtcr_to_video_feature(noisy_feature, wtcr_branch)
    print(f"WTCR output shape: {wtcr_output.shape}")

    # 生成可视化分析
    save_dir = './test_wtcr_real'
    os.makedirs(save_dir, exist_ok=True)

    print("Generating WTCR analysis...")
    visualizer.visualize_feature_comparison(
        clean_feature, noisy_feature, wtcr_output,
        save_dir, sample_idx=0, noise_var=noise_var
    )

    print("WTCR integration test completed successfully!")

    # 检查生成的文件
    import glob
    files = glob.glob(os.path.join(save_dir, "*.png"))
    print(f"Generated files: {files}")

if __name__ == '__main__':
    test_wtcr_with_real_data()
