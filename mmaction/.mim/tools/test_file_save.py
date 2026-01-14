#!/usr/bin/env python3
"""
测试文件保存功能
"""

import torch
import numpy as np
from pathlib import Path
import cv2
import os

def test_file_save():
    """测试文件保存"""
    # 创建测试目录
    test_dir = Path('./test_save_dir')
    test_dir.mkdir(exist_ok=True)
    print(f"Created test directory: {test_dir}")
    print(f"Directory exists: {test_dir.exists()}")

    # 创建一个简单的图像
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_path = test_dir / 'test_image.png'

    # 保存图像
    cv2.imwrite(str(test_path), test_image)
    print(f"Saved test image to: {test_path}")
    print(f"File exists: {test_path.exists()}")

    if test_path.exists():
        print(f"File size: {test_path.stat().st_size} bytes")

        # 列出目录内容
        print("Directory contents:")
        for item in test_dir.iterdir():
            print(f"  {item}")

    # 清理
    if test_path.exists():
        test_path.unlink()
    test_dir.rmdir()
    print("Cleaned up test files")

if __name__ == '__main__':
    test_file_save()
