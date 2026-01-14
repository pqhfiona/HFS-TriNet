import os
import json
import math
import shutil
from pathlib import Path

import torch
import numpy as np
import cv2
from mmengine.config import Config
from mmengine.runner import Runner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入WTCR可视化工具
from wtcr_visualization import WTCRVisualizer


# 临时 monkey-patch：确保 AccMetric 存在 current_epoch 属性，避免评估时 hook 访问出错
try:
    from mmaction.evaluation.metrics import AccMetric as _AccMetric
    if not hasattr(_AccMetric, 'current_epoch'):
        _AccMetric.current_epoch = 0
    if not hasattr(_AccMetric, 'work_dir'):
        _AccMetric.work_dir = ''
except Exception:
    try:
        # 某些版本的 mmaction 将 metrics 放在其它位置，忽略失败
        pass
    except Exception:
        pass

# 程序化批量评估并保存结果与可视化示例
cfg_path = '/data/pqh/env/MM/configs/recognition/my_work/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py'
noise_vars = [0.01, 0.02, 0.05, 0.1, 0.2]
repeats_per_var = 3
output_root = Path('./eval_speckle_results')
visualize_num_clips = 1  # 每次保存多少个样本的可视化对比

# smoke test 开关：用于快速验证（默认开启以避免长时间跑全量实验）
smoke_test = False
if smoke_test:
    noise_vars = [0.05]
    repeats_per_var = 1

# WTCR可视化分析开关
enable_wtcr_analysis = True  # 启用WTCR特征域可视化分析

def apply_speckle_to_tensor(x: torch.Tensor, var: float, device=None):
    """对输入张量做乘性 speckle： x -> x + x * N(0, sqrt(var))"""
    if device is None:
        device = x.device
    noise = torch.randn_like(x, device=device) * math.sqrt(var)
    return x + x * noise

def save_frame_grid(orig_frames, noisy_frames, denoised_frames, out_path):
    # orig_frames 等为 list of HWC uint8
    # 拼接为一张图：每一行为 orig / noisy / denoised
    rows = []
    for o, n, d in zip(orig_frames, noisy_frames, denoised_frames):
        row = np.concatenate([o, n, d], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), grid)

def denoise_frames_cv2(frames):
    denoised = []
    for f in frames:
        # 使用 fastNlMeansDenoisingColored（对于单通道也能工作）
        if f.ndim == 2:
            den = cv2.fastNlMeansDenoising(f, None, 10, 7, 21)
        else:
            den = cv2.fastNlMeansDenoisingColored(f, None, 10, 10, 7, 21)
        denoised.append(den)
    return denoised

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
        except Exception:
            try:
                storage_list.append(output.cpu())
            except Exception:
                storage_list.append(output)

    handle = module.register_forward_hook(hook)
    return handle

def compute_log_spectrum(frame):
    """Compute log-power 2D FFT spectrum for an HWC uint8 image."""
    import numpy as _np
    if isinstance(frame, (list, tuple)):
        frame = frame[0]
    arr = _np.asarray(frame)
    if arr.ndim == 3:
        # convert to gray by channel mean
        gray = arr.mean(axis=2)
    else:
        gray = arr
    f = _np.fft.fft2(gray.astype('float32'))
    fshift = _np.fft.fftshift(f)
    magnitude = _np.abs(fshift)
    log_mag = _np.log1p(magnitude)
    # normalize to 0-1
    log_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
    return (log_mag * 255).astype('uint8')

def save_feature_and_spectrum_grid(out_path, orig_frames, noisy_frames, denoised_frames, feature_tensor, max_frames=3):
    """
    Save a grid image containing for up to max_frames:
      [orig_img | orig_spectrum | noisy_img | noisy_spectrum | denoised_img | denoised_spectrum | feat_map]
    feature_tensor: CPU tensor with shape (N,C,T,H,W) or (C,T,H,W) or (T,H,W)
    """
    import numpy as _np
    # prepare feature map: reduce channels -> compute channel-mean projection
    feat = feature_tensor
    if isinstance(feat, list):
        feat = feat[0]
    arr = _np.array(feat)
    # handle shapes
    if arr.ndim == 5:
        # N,C,T,H,W -> take first sample
        arr = arr[0]
    if arr.ndim == 4:
        # C,T,H,W -> compute channel mean -> T,H,W
        arr = arr.mean(axis=0)
    # arr now T,H,W or H,W
    if arr.ndim == 2:
        arr = arr[None, ...]

    rows = []
    F = min(max_frames, arr.shape[0], len(orig_frames))
    for t in range(F):
        o = orig_frames[t]
        n = noisy_frames[t]
        d = denoised_frames[t]
        o_spec = compute_log_spectrum(o)
        n_spec = compute_log_spectrum(n)
        d_spec = compute_log_spectrum(d)
        # feature map for time t
        feat_map = arr[t]
        # normalize feature map to 0-255
        fm = feat_map.astype('float32')
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        fm = (fm * 255).astype('uint8')
        # resize all to same height
        H = 128
        def to_rgb(img):
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        o_r = cv2.resize(o, (H, H))
        n_r = cv2.resize(n, (H, H))
        d_r = cv2.resize(d, (H, H))
        o_s = cv2.resize(o_spec, (H, H))
        n_s = cv2.resize(n_spec, (H, H))
        d_s = cv2.resize(d_spec, (H, H))
        fm_r = cv2.resize(fm, (H, H))
        fm_r = cv2.applyColorMap(fm_r, cv2.COLORMAP_JET)

        left = _np.concatenate([to_rgb(o_r), to_rgb(o_s), to_rgb(n_r), to_rgb(n_s), to_rgb(d_r), to_rgb(d_s)], axis=1)
        right = fm_r
        row = _np.concatenate([left, right], axis=1)
        rows.append(row)

    grid = _np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), grid)

def tensor_to_frames(x: torch.Tensor):
    # x: (C,T,H,W) or (N,C,T,H,W) -> return list of HWC uint8 for first sample
    arr = x.detach().cpu().numpy()
    if arr.ndim == 5:
        arr = arr[0]  # N=1
    # arr: C,T,H,W
    C, T, H, W = arr.shape
    frames = []
    for t in range(T):
        frame = arr[:, t, :, :]  # C,H,W
        frame = np.transpose(frame, (1, 2, 0))  # H,W,C
        # assume input in [0,1] or normalized - try to rescale
        if frame.dtype != np.uint8:
            f = frame
            f = np.clip(f, 0.0, 1.0)
            frame = (f * 255.0).astype(np.uint8)
        frames.append(frame)
    return frames


def make_serializable(obj):
    """Recursively convert numpy/torch types to native Python types for JSON."""
    # numpy scalar types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # torch tensors
    try:
        import torch as _torch
        if isinstance(obj, _torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    # dict / list / tuple
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    # fallback: try to cast floats
    if isinstance(obj, float):
        return float(obj)
    return obj

def run_eval():
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
results_summary = {}

for var in noise_vars:
        var_dir = output_root / f'var_{var:.3f}'
        var_dir.mkdir(parents=True, exist_ok=True)
        metrics_list = []
        for r in range(repeats_per_var):
            cfg = Config.fromfile(cfg_path)
            # ensure work_dir exists for Runner
            if cfg.get('work_dir', None) is None:
                cfg.work_dir = str(var_dir / f'run_{r}')

            # Load checkpoint - modify config to load trained weights
            checkpoint_path = '/data/pqh/env/MM/checkpoints/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-97d0835d.pth'
            if not cfg.get('load_from', None):
                cfg.load_from = checkpoint_path
            pass

            runner = Runner.from_cfg(cfg)

            # 确保 evaluator/metrics 上存在 current_epoch 与 work_dir，避免 PredictionSaveHook 或 AccMetric 访问失败
            try:
                evaluator = None
                if hasattr(runner, 'val_loop') and hasattr(runner.val_loop, 'evaluator'):
                    evaluator = runner.val_loop.evaluator
                elif hasattr(runner, 'test_loop') and hasattr(runner.test_loop, 'evaluator'):
                    evaluator = runner.test_loop.evaluator
                else:
                    evaluator = getattr(runner, 'evaluator', None)

                if evaluator is not None:
                    metrics = getattr(evaluator, 'metrics', None)
                    if metrics:
                        for metric in metrics:
                            if not hasattr(metric, 'current_epoch'):
                                setattr(metric, 'current_epoch', getattr(runner, 'epoch', 0))
                            if not hasattr(metric, 'work_dir'):
                                setattr(metric, 'work_dir', cfg.get('work_dir', ''))
            except Exception:
                # 若定位不到 evaluator，不阻塞主流程，交由 hook 自身处理
                pass

            # monkey-patch model.forward，在进入模型前加入乘性噪声（仅用于评估）
            model = runner.model
            target = model.module if hasattr(model, 'module') else model
            orig_forward = target.forward

            def noisy_forward(inputs, *args, **kwargs):
                # inputs 可能是 Tensor 或 list/tuple 包含 Tensor
                try:
                    if isinstance(inputs, torch.Tensor):
                        noisy_in = apply_speckle_to_tensor(inputs, var, device=inputs.device)
                        return orig_forward(noisy_in, *args, **kwargs)
                    elif isinstance(inputs, (list, tuple)):
                        new_inputs = []
                        for item in inputs:
                            if isinstance(item, torch.Tensor):
                                new_inputs.append(apply_speckle_to_tensor(item, var, device=item.device))
                            else:
                                new_inputs.append(item)
                        return orig_forward(new_inputs, *args, **kwargs)
                    else:
                        return orig_forward(inputs, *args, **kwargs)
                except Exception:
                    return orig_forward(inputs, *args, **kwargs)

            target.forward = noisy_forward

            # 执行评估（runner.test/validate 依据你的 Runner API）
            try:
                metrics = runner.test()
            except Exception as e:
                # 如果 runner.test 不可用，尝试 runner.validate
                try:
                    metrics = runner.validate()
                except Exception as e2:
                    metrics = {'error': str(e)[:200]}

            metrics_list.append(metrics)

            # 可视化：创建简单的测试样本进行分析
            print("Starting visualization block...")
            try:
                print("Creating test sample for visualization...")
                # 模拟一个视频输入: (1, 3, 8, 224, 224) - 1 batch, 3 channels, 8 frames, 224x224
                device = next(runner.model.parameters()).device
                print(f"Using device: {device}")
                test_input = torch.randn(1, 3, 8, 224, 224, device=device)
                print(f"Created test input with shape: {test_input.shape}")

                # 获取原始帧和加噪帧
                orig_frames = tensor_to_frames(test_input)
                noisy_tensor = apply_speckle_to_tensor(test_input.clone(), var, device=device)
                noisy_frames = tensor_to_frames(noisy_tensor)
                denoised_frames = denoise_frames_cv2(noisy_frames)
                out_path = var_dir / f'visual_sample_{r}_test.png'
                print(f"Attempting to save to: {out_path}")
                print(f"var_dir exists: {var_dir.exists()}")
                print(f"var_dir path: {var_dir}")
                try:
                    save_frame_grid(orig_frames, noisy_frames, denoised_frames, out_path)
                    print(f"Saved image visualization to {out_path}")
                    print(f"File exists after save: {out_path.exists()}")
                    if out_path.exists():
                        print(f"File size: {out_path.stat().st_size} bytes")
                    else:
                        print("ERROR: File was not actually saved!")
                except Exception as save_error:
                    print(f"Save failed with error: {save_error}")
                    import traceback
                    traceback.print_exc()

                # 特征与频谱可视化（尝试在模型上注册 hook 并保存特征对比）
                try:
                    features = []
                    layer_name = 'backbone.layer4'  # TPN backbone的layer4
                    target_model = runner.model.module if hasattr(runner.model, 'module') else runner.model
                    handle = register_hook_by_name(target_model, layer_name, features)
                    target_model.eval()
                    with torch.no_grad():
                        # 直接调用backbone，避免neck的问题
                        backbone_output = target_model.backbone(test_input)
                        if isinstance(backbone_output, tuple):
                            # 使用最后一个特征（通常是最高层的特征）
                            layer4_output = backbone_output[-1]
                        else:
                            layer4_output = backbone_output
                    handle.remove()
                    print(f"Hook captured {len(features)} feature tensors")
                    if features:
                        print(f"Feature tensor shape: {features[0].shape}")
                        feat_out = var_dir / f'feature_sample_{r}_test.png'
                        save_feature_and_spectrum_grid(feat_out, orig_frames, noisy_frames, denoised_frames, features[0])
                        print(f"Saved feature visualization to {feat_out}")
                        print(f"WTCR analysis enabled: {enable_wtcr_analysis}")

                        # 添加WTCR特征域可视化分析
                        if enable_wtcr_analysis:
                            print("Starting WTCR analysis...")
                            try:
                                print("Creating WTCR visualizer...")
                                wtcr_visualizer = WTCRVisualizer(wavelet_type='db1', wt_levels=1)
                                print("WTCR visualizer created successfully")

                                print("Creating clean features...")
                                # 为clean输入创建特征
                                clean_features = []
                                clean_handle = register_hook_by_name(target_model, layer_name, clean_features)
                                with torch.no_grad():
                                    backbone_output = target_model.backbone(test_input)
                                    if isinstance(backbone_output, tuple):
                                        clean_layer4_output = backbone_output[-1]
                                    else:
                                        clean_layer4_output = backbone_output
                                clean_handle.remove()
                                print(f"Clean features captured: {len(clean_features)}")

                                if clean_features:
                                    print("Clean features available, proceeding with WTCR analysis")
                                else:
                                    print("No clean features captured, using backbone output")
                                    clean_features = [clean_layer4_output]

                                if clean_features:
                                    clean_feature = clean_features[0].detach()
                                    print(f"Clean feature shape: {clean_feature.shape}")

                                    print("Creating noisy features...")
                                    # 重新获取加噪特征
                                    noisy_features_hook = []
                                    noisy_handle = register_hook_by_name(target_model, layer_name, noisy_features_hook)
                                    with torch.no_grad():
                                        noisy_backbone_output = target_model.backbone(noisy_tensor)
                                        if isinstance(noisy_backbone_output, tuple):
                                            noisy_layer4_output = noisy_backbone_output[-1]
                                        else:
                                            noisy_layer4_output = noisy_backbone_output
                                    noisy_handle.remove()
                                    print(f"Noisy features captured: {len(noisy_features_hook)}")

                                    if noisy_features_hook:
                                        noisy_feature = noisy_features_hook[0].detach()
                                    else:
                                        # 如果hook没有工作，使用直接的backbone输出
                                        noisy_feature = noisy_layer4_output.detach()
                                        print("Using backbone output for noisy features")

                                    print(f"Noisy feature shape: {noisy_feature.shape}")

                                    print("Creating WTCR branch...")
                                    # 创建WTCR分支并处理加噪特征
                                    feature_channels = clean_feature.shape[1]
                                    wtcr_branch = wtcr_visualizer.create_wtcr_branch(
                                        in_channels=feature_channels,
                                        out_channels=feature_channels
                                    ).to(clean_feature.device)
                                    print("WTCR branch created")

                                    print("Applying WTCR to video features...")
                                    # 应用WTCR到视频特征
                                    wtcr_output = wtcr_visualizer.apply_wtcr_to_video_feature(noisy_feature, wtcr_branch)
                                    print(f"WTCR output shape: {wtcr_output.shape}")

                                    print("Generating WTCR visualizations...")
                                    # 生成WTCR可视化分析
                                    wtcr_analysis_dir = var_dir / 'wtcr_analysis'
                                    wtcr_visualizer.visualize_feature_comparison(
                                        clean_feature, noisy_feature, wtcr_output,
                                        str(wtcr_analysis_dir), sample_idx=0, noise_var=var
                                    )

                                    print(f"WTCR analysis completed for test sample, noise_var {var}")

                            except Exception as wtcr_error:
                                print(f"WTCR visualization failed: {str(wtcr_error)}")
                                import traceback
                                traceback.print_exc()

                except Exception as hook_error:
                    print(f"Hook visualization failed: {str(hook_error)}")
            except StopIteration:
                pass
            except Exception as vis_error:
                print(f"Visualization failed with error: {str(vis_error)}")
                import traceback
                traceback.print_exc()

            # 恢复 forward
            target.forward = orig_forward

        # 汇总该强度的 metrics（若为 dict，计算各字段均值）
        aggregate = {}
        try:
            # 假设 metrics_list 为 dict 列表，取数值字段的平均
            numeric_keys = set()
            for m in metrics_list:
                if isinstance(m, dict):
                    for k, v in m.items():
                        if isinstance(v, (int, float)):
                            numeric_keys.add(k)
            for k in numeric_keys:
                vals = [m.get(k, float('nan')) for m in metrics_list]
                vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
                if vals:
                    aggregate[k] = float(np.mean(vals))
        except Exception:
            aggregate = {'raw': metrics_list}

        results_summary[var] = {'per_repeat': metrics_list, 'aggregate': aggregate}
        serial = make_serializable(results_summary[var])
        with open(var_dir / 'metrics_summary.json', 'w') as f:
            json.dump(serial, f, indent=2)

# 总结果保存
# make entire summary JSON-serializable
serial_all = make_serializable(results_summary)
with open(output_root / 'all_results_summary.json', 'w') as f:
    json.dump(serial_all, f, indent=2)

if __name__ == '__main__':
    run_eval()