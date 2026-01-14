#!/usr/bin/env python3
"""
测试hook功能
"""

import torch
import sys
sys.path.append('/data/pqh/env/MM')

from mmengine.config import Config
from mmengine.runner import Runner

def register_hook_by_name(model, layer_name, storage_list):
    """Register a forward hook on module specified by dotted layer_name (e.g. 'backbone.layer4')."""
    module = model
    for attr in layer_name.split('.'):
        if hasattr(module, attr):
            module = getattr(module, attr)
        else:
            raise AttributeError(f"Module has no attribute {attr} when resolving {layer_name}")

    def hook(module, input, output):
        # store output (detach on CPU)
        try:
            storage_list.append(output.detach().cpu())
            print(f"Hook triggered! Output shape: {output.shape}")
        except Exception as e:
            print(f"Hook error: {e}")
            try:
                storage_list.append(output.cpu())
            except Exception:
                storage_list.append(output)

    handle = module.register_forward_hook(hook)
    return handle

def test_hook():
    """测试hook功能"""
    print("Loading config and model...")

    cfg_path = '/data/pqh/env/MM/configs/recognition/my_work/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py'
    cfg = Config.fromfile(cfg_path)

    # 设置work_dir和checkpoint
    cfg.work_dir = './temp_check'
    checkpoint_path = '/data/pqh/env/MM/checkpoints/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-97d0835d.pth'
    cfg.load_from = checkpoint_path

    runner = Runner.from_cfg(cfg)
    model = runner.model

    print("Model loaded successfully")

    # 创建测试输入 - backbone期望的格式是(N, C, T, H, W)
    device = next(model.parameters()).device
    # 对于单个clip，格式是 (N, C, T, H, W)
    test_input = torch.randn(1, 3, 8, 224, 224, device=device)
    print(f"Test input shape: {test_input.shape}")

    # 注册hook
    features = []
    layer_name = 'backbone.layer4'
    print(f"Registering hook on {layer_name}")

    try:
        handle = register_hook_by_name(model, layer_name, features)
        print("Hook registered successfully")
    except Exception as e:
        print(f"Hook registration failed: {e}")
        return

    # 运行推理
    model.eval()
    print("Running inference...")
    with torch.no_grad():
        try:
            # 直接使用backbone输出，避免neck的问题
            backbone_output = model.backbone(test_input)
            print(f"Backbone output type: {type(backbone_output)}")
            if isinstance(backbone_output, tuple):
                for i, feat in enumerate(backbone_output):
                    print(f"Backbone feature {i} shape: {feat.shape}")
                # 使用最后一个特征（通常是最高层的特征）
                output = backbone_output[-1]  # layer4的输出
                print(f"Using backbone feature shape: {output.shape}")
            else:
                output = backbone_output
                print(f"Backbone output shape: {output.shape}")
        except Exception as e:
            print(f"Backbone inference failed: {e}")
            return

    # 移除hook
    handle.remove()

    print(f"Hook captured {len(features)} features")
    if features:
        print(f"Feature shape: {features[0].shape}")

if __name__ == '__main__':
    test_hook()
