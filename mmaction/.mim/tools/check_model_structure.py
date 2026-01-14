#!/usr/bin/env python3
"""
检查模型结构，找到合适的hook点
"""

import torch
import sys
sys.path.append('/data/pqh/env/MM')

from mmengine.config import Config
from mmengine.runner import Runner

def check_model_structure():
    """检查模型结构"""
    print("Loading config and model...")

    cfg_path = '/data/pqh/env/MM/configs/recognition/my_work/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py'
    cfg = Config.fromfile(cfg_path)

    # 设置work_dir和checkpoint
    cfg.work_dir = './temp_check'
    checkpoint_path = '/data/pqh/env/MM/checkpoints/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-97d0835d.pth'
    cfg.load_from = checkpoint_path

    runner = Runner.from_cfg(cfg)
    model = runner.model

    print("Model structure:")
    print(model)

    print("\nNamed modules:")
    for name, module in model.named_modules():
        print(f"  {name}: {type(module).__name__}")

    print("\nChecking backbone structure:")
    if hasattr(model, 'backbone'):
        print("Backbone modules:")
        for name, module in model.backbone.named_modules():
            print(f"  backbone.{name}: {type(module).__name__}")

    if hasattr(model, 'neck'):
        print("Neck modules:")
        for name, module in model.neck.named_modules():
            print(f"  neck.{name}: {type(module).__name__}")

if __name__ == '__main__':
    check_model_structure()
