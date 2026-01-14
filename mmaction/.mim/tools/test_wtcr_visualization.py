#!/usr/bin/env python3
"""
测试WTCR可视化功能
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# 添加路径
sys.path.append('/data/pqh/env/MM/tools')

from wtcr_visualization import WTCRVisualizer

def test_wtcr_visualization():
    """测试WTCR可视化功能"""
    print("Testing WTCR visualization...")

    # 创建可视化器
    visualizer = WTCRVisualizer(wavelet_type='db1', wt_levels=1)

    # 创建WTCR分支
    in_channels = 64
    out_channels = 64
    wtcr_branch = visualizer.create_wtcr_branch(in_channels, out_channels)

    # 生成测试特征数据
    batch_size, height, width = 2, 32, 32
    clean_feature = torch.randn(batch_size, in_channels, height, width)

    # 添加speckle噪声
    noise_var = 0.05
    noisy_feature = visualizer.apply_speckle_noise(clean_feature, noise_var)

    # WTCR处理
    wtcr_branch.eval()
    with torch.no_grad():
        wtcr_output = wtcr_branch(noisy_feature)

    print(f"Clean feature shape: {clean_feature.shape}")
    print(f"Noisy feature shape: {noisy_feature.shape}")
    print(f"WTCR output shape: {wtcr_output.shape}")

    # 生成可视化分析
    save_dir = './test_wtcr_analysis'
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving analysis to: {save_dir}")
    visualizer.visualize_feature_comparison(
        clean_feature, noisy_feature, wtcr_output,
        save_dir, sample_idx=0, noise_var=noise_var
    )

    print("WTCR visualization test completed successfully!")

    # 检查生成的文件
    import glob
    files = glob.glob(os.path.join(save_dir, "*.png"))
    print(f"Generated files: {files}")

if __name__ == '__main__':
    test_wtcr_visualization()
