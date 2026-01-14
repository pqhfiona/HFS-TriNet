import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any
from mmengine.config import Config
from mmengine.runner import Runner
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# 设置matplotlib后端
plt.switch_backend('Agg')

# 导入WTCR可视化工具
from wtcr_visualization import WTCRVisualizer


def generate_gamma_speckle_noise(shape: torch.Size, k: float, device: torch.device) -> torch.Tensor:
    """
    生成Gamma分布的speckle噪声（乘性噪声）

    Args:
        shape: 噪声张量的形状
        k: Gamma分布的形状参数（scale parameter）
        device: 设备

    Returns:
        Gamma分布的噪声张量
    """
    # Gamma分布：使用形状参数k，尺度参数1/k
    # 期望值为1，方差为1/k
    gamma_dist = torch.distributions.Gamma(k, 1.0/k)
    gamma_noise = gamma_dist.sample(shape).to(device)
    return gamma_noise


def apply_speckle_noise_gamma(inputs: torch.Tensor, k: float) -> torch.Tensor:
    """
    对输入张量应用Gamma分布的speckle噪声（乘性噪声）

    Args:
        inputs: 输入张量 (N, C, T, H, W) 或其他形状
        k: Gamma分布的形状参数，越小噪声越强

    Returns:
        加噪后的张量
    """
    if k == float('inf'):  # Clean case
        return inputs.clone()

    device = inputs.device
    noise = generate_gamma_speckle_noise(inputs.shape, k, device)
    # 乘性噪声：x * noise，其中noise ~ Gamma(k, 1/k)
    noisy_inputs = inputs * noise
    return noisy_inputs


def get_noise_configs() -> List[Dict[str, Any]]:
    """
    获取预定义的噪声配置

    Returns:
        噪声配置列表
    """
    return [
        {'name': 'clean', 'k': float('inf'), 'description': '无噪声'},
        {'name': 'low', 'k': 10.0, 'description': '低噪声强度 (k=10)'},
        {'name': 'medium', 'k': 5.0, 'description': '中噪声强度 (k=5)'},
        {'name': 'high', 'k': 2.0, 'description': '高噪声强度 (k=2)'}
    ]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    计算分类性能指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（用于AUC计算）

    Returns:
        性能指标字典
    """
    metrics = {}

    # 基本指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)

    # 特异性 (Specificity) = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 即recall
        metrics['positive_accuracy'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['negative_accuracy'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC (如果有概率值)
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0  # 处理AUC计算失败的情况

    return metrics


def create_baseline_model(cfg: Config, model_type: str) -> Config:
    """
    创建baseline模型配置（不含WTCR或不含SAM特征）

    Args:
        cfg: 原始配置
        model_type: baseline类型 ('no_wtcr', 'no_sam')

    Returns:
        修改后的配置
    """
    # 复制配置以避免修改原始配置
    baseline_cfg = cfg.clone()

    if model_type == 'no_wtcr':
        # 移除WTCR分支 - 这里需要根据具体实现修改
        # 由于WTCR可能在neck或自定义模块中实现，这里暂时保持原样
        # 实际使用时需要根据模型的具体实现来移除WTCR相关组件
        print("警告: no_wtcr baseline暂未实现，将使用完整模型")
        pass

    elif model_type == 'no_sam':
        # 移除SAM特征 - 这里需要根据具体实现修改
        # 由于SAM特征可能在数据预处理或特征提取中，这里暂时保持原样
        # 实际使用时需要根据模型的具体实现来移除SAM相关组件
        print("警告: no_sam baseline暂未实现，将使用完整模型")
        pass

    return baseline_cfg


def evaluate_model_with_noise(runner: Runner, noise_k: float, save_predictions: bool = False) -> Dict[str, Any]:
    """
    在指定噪声强度下评估模型

    Args:
        runner: MMEngine Runner对象
        noise_k: 噪声强度参数 (Gamma分布的k值)
        save_predictions: 是否保存预测结果

    Returns:
        评估结果字典
    """
    model = runner.model
    original_forward = model.forward

    # 用于收集预测结果的全局变量
    collected_data = {
        'labels': [],
        'predictions': [],
        'probabilities': []
    }

    def noisy_forward(inputs, data_samples=None, **kwargs):
        """修改forward函数以添加噪声"""
        if isinstance(inputs, torch.Tensor) and noise_k != float('inf'):
            # 应用Gamma speckle噪声
            inputs = apply_speckle_noise_gamma(inputs, noise_k)

        # 调用原始forward
        outputs = original_forward(inputs, data_samples=data_samples, **kwargs)

        return outputs

    # 创建hook来收集预测结果
    def prediction_hook(module, inputs, outputs):
        """Hook函数用于收集预测结果"""
        if save_predictions and hasattr(outputs, 'pred_score'):
            pred_scores = outputs.pred_score
            if isinstance(pred_scores, torch.Tensor):
                # 获取预测概率
                probs = torch.softmax(pred_scores, dim=-1)
                collected_data['probabilities'].extend(probs[:, 1].cpu().numpy())

                # 获取预测标签
                preds = torch.argmax(pred_scores, dim=-1).cpu().numpy()
                collected_data['predictions'].extend(preds)

        # 收集真实标签
        if hasattr(outputs, 'data_samples'):
            for sample in outputs.data_samples:
                if hasattr(sample, 'gt_label'):
                    collected_data['labels'].append(sample.gt_label.item())

    # 注册hook到模型的分类头
    hooks = []
    for name, module in model.named_modules():
        if 'head' in name.lower() or 'cls' in name.lower() or 'fc' in name.lower():
            hook = module.register_forward_hook(prediction_hook)
            hooks.append(hook)
            break  # 只hook第一个找到的分类相关模块

    # 替换模型的forward函数
    model.forward = noisy_forward

    try:
        # 执行评估
        metrics = runner.test()

        # 计算额外的指标
        if collected_data['labels'] and collected_data['predictions']:
            y_true = np.array(collected_data['labels'])
            y_pred = np.array(collected_data['predictions'])
            y_prob = np.array(collected_data['probabilities']) if collected_data['probabilities'] else None

            custom_metrics = compute_metrics(y_true, y_pred, y_prob)
            metrics.update(custom_metrics)

            if save_predictions:
                metrics['custom_metrics'] = collected_data

    finally:
        # 移除hooks
        for hook in hooks:
            hook.remove()

        # 恢复原始forward函数
        model.forward = original_forward

    return metrics


def plot_performance_curves(results: Dict[str, Any], save_path: str):
    """
    绘制性能随噪声强度变化的曲线图

    Args:
        results: 评估结果字典，格式为 {model_type: {noise_name: {'avg_metrics': {...}}}}
        save_path: 保存路径
    """
    # 获取所有噪声配置
    noise_configs = get_noise_configs()
    noise_order = [config['name'] for config in noise_configs]
    noise_labels = [config['description'] for config in noise_configs]

    # 提取指标
    metrics_to_plot = ['accuracy', 'auc', 'f1_score', 'precision', 'recall', 'specificity']

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o-', 's-', '^-', 'd-', '*-']

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        for i, (model_type, model_results) in enumerate(results.items()):
            if model_type not in ['full', 'no_wtcr', 'no_sam']:
                continue

            metric_values = []
            for noise_name in noise_order:
                if noise_name in model_results and 'avg_metrics' in model_results[noise_name]:
                    avg_metrics = model_results[noise_name]['avg_metrics']
                    if metric in avg_metrics:
                        metric_values.append(avg_metrics[metric])
                    else:
                        metric_values.append(None)
                else:
                    metric_values.append(None)

            # 过滤掉None值
            valid_values = [(j, v) for j, v in enumerate(metric_values) if v is not None]
            if valid_values:
                x_vals = [j for j, v in valid_values]
                y_vals = [v for j, v in valid_values]

                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]

                ax.plot(x_vals, y_vals, marker, color=color, linewidth=2,
                       markersize=8, label=model_type, alpha=0.8)

        ax.set_xlabel('Noise Intensity', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} vs Noise Intensity', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(noise_labels)))
        ax.set_xticklabels(noise_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 设置y轴范围
        if metric in ['accuracy', 'auc', 'f1_score', 'precision', 'recall', 'specificity']:
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"性能曲线图已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='鲁棒性评估实验')
    parser.add_argument('--config', default='/data/pqh/env/MM/configs/recognition/my_work/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', default='/data/pqh/env/MM/checkpoints/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-97d0835d.pth',
                       help='模型checkpoint路径')
    parser.add_argument('--work-dir', default='./robustness_eval_results',
                       help='结果保存目录')
    parser.add_argument('--noise-configs', nargs='+',
                       default=['clean', 'low', 'medium', 'high'],
                       help='噪声配置：clean(无噪声), low(k=10), medium(k=5), high(k=2)')
    parser.add_argument('--repeats', type=int, default=3,
                       help='每个噪声强度重复次数')
    parser.add_argument('--baseline-types', nargs='+', default=['full', 'no_wtcr', 'no_sam'],
                       help='要评估的模型类型：full(完整模型), no_wtcr(无WTCR), no_sam(无SAM)')

    args = parser.parse_args()

    # 创建结果目录
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 解析噪声配置
    all_noise_configs = get_noise_configs()
    noise_config_dict = {config['name']: config for config in all_noise_configs}

    noise_configs = []
    for config_name in args.noise_configs:
        if config_name in noise_config_dict:
            noise_configs.append(noise_config_dict[config_name])
        else:
            print(f"警告: 未知的噪声配置 '{config_name}'，跳过")

    print(f"噪声配置: {noise_configs}")
    print(f"基线类型: {args.baseline_types}")

    # 存储所有结果
    all_results = {}

    # 对每个模型类型进行评估
    for model_type in args.baseline_types:
        print(f"\n=== 评估模型类型: {model_type} ===")

        model_results = {}

        # 对每个噪声强度进行评估
        for noise_config in noise_configs:
            print(f"\n评估噪声强度: {noise_config['name']} (k={noise_config['k']})")

            noise_results = []
            noise_k = noise_config['k']

            # 重复评估
            for repeat in range(args.repeats):
                print(f"  重复 {repeat + 1}/{args.repeats}")

                # 加载配置和模型
                cfg = Config.fromfile(args.config)
                cfg.work_dir = str(work_dir / model_type / noise_config['name'] / f'repeat_{repeat}')

                if model_type == 'full':
                    # 使用完整模型
                    pass
                else:
                    # 创建baseline模型配置
                    cfg = create_baseline_model(cfg, model_type)

                # 加载checkpoint
                if args.checkpoint and os.path.exists(args.checkpoint):
                    cfg.load_from = args.checkpoint

                # 创建runner
                runner = Runner.from_cfg(cfg)

                # 执行评估
                metrics = evaluate_model_with_noise(runner, noise_k, save_predictions=True)
                noise_results.append(metrics)

            # 计算平均指标
            avg_metrics = {}
            for key in noise_results[0].keys():
                if key != 'custom_metrics' and isinstance(noise_results[0][key], (int, float)):
                    values = [r[key] for r in noise_results if key in r and isinstance(r[key], (int, float))]
                    if values:
                        avg_metrics[key] = np.mean(values)

            model_results[noise_config['name']] = {
                'avg_metrics': avg_metrics,
                'per_repeat': noise_results
            }

            # 保存该噪声强度下的结果
            noise_dir = work_dir / model_type / noise_config['name']
            noise_dir.mkdir(parents=True, exist_ok=True)

            with open(noise_dir / 'results.json', 'w') as f:
                json.dump({
                    'noise_config': noise_config,
                    'avg_metrics': avg_metrics,
                    'per_repeat': noise_results
                }, f, indent=2)

        all_results[model_type] = model_results

    # 保存总结果
    with open(work_dir / 'all_results.json', 'w') as f:
        # 转换为可序列化的格式
        serializable_results = {}
        for model_type, model_res in all_results.items():
            serializable_results[model_type] = {}
            for noise_name, noise_res in model_res.items():
                serializable_results[model_type][noise_name] = {
                    'avg_metrics': noise_res['avg_metrics'],
                    'per_repeat_count': len(noise_res['per_repeat'])
                }
        json.dump(serializable_results, f, indent=2)

    # 生成性能曲线图
    try:
        plot_save_path = work_dir / 'performance_curves.png'
        plot_performance_curves(all_results, str(plot_save_path))
        print(f"\n性能曲线图已保存到: {plot_save_path}")
    except Exception as e:
        print(f"生成性能曲线图失败: {e}")

    # 调用统一的评估函数
    run_robustness_eval(args)


def run_robustness_eval(args):
    """
    运行鲁棒性评估（供外部调用）

    Args:
        args: 参数对象，包含config, checkpoint, work_dir等属性
    """
    # 创建结果目录
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 解析噪声配置
    all_noise_configs = get_noise_configs()
    noise_config_dict = {config['name']: config for config in all_noise_configs}

    noise_configs = []
    for config_name in args.noise_configs:
        if config_name in noise_config_dict:
            noise_configs.append(noise_config_dict[config_name])
        else:
            print(f"警告: 未知的噪声配置 '{config_name}'，跳过")

    print(f"噪声配置: {noise_configs}")
    print(f"基线类型: {args.baseline_types}")

    # 存储所有结果
    all_results = {}

    # 对每个模型类型进行评估
    for model_type in args.baseline_types:
        print(f"\n=== 评估模型类型: {model_type} ===")

        model_results = {}

        # 对每个噪声强度进行评估
        for noise_config in noise_configs:
            print(f"\n评估噪声强度: {noise_config['name']} (k={noise_config['k']})")

            noise_results = []
            noise_k = noise_config['k']

            # 重复评估
            for repeat in range(args.repeats):
                print(f"  重复 {repeat + 1}/{args.repeats}")

                # 加载配置和模型
                cfg = Config.fromfile(args.config)
                cfg.work_dir = str(work_dir / model_type / noise_config['name'] / f'repeat_{repeat}')

                if model_type == 'full':
                    # 使用完整模型
                    pass
                else:
                    # 创建baseline模型配置
                    cfg = create_baseline_model(cfg, model_type)

                # 加载checkpoint
                if args.checkpoint and os.path.exists(args.checkpoint):
                    cfg.load_from = args.checkpoint

                # 创建runner
                runner = Runner.from_cfg(cfg)

                # 执行评估
                metrics = evaluate_model_with_noise(runner, noise_k, save_predictions=True)
                noise_results.append(metrics)

            # 计算平均指标
            avg_metrics = {}
            for key in noise_results[0].keys():
                if key != 'custom_metrics' and isinstance(noise_results[0][key], (int, float)):
                    values = [r[key] for r in noise_results if key in r and isinstance(r[key], (int, float))]
                    if values:
                        avg_metrics[key] = np.mean(values)

            model_results[noise_config['name']] = {
                'avg_metrics': avg_metrics,
                'per_repeat': noise_results
            }

            # 保存该噪声强度下的结果
            noise_dir = work_dir / model_type / noise_config['name']
            noise_dir.mkdir(parents=True, exist_ok=True)

            with open(noise_dir / 'results.json', 'w') as f:
                json.dump({
                    'noise_config': noise_config,
                    'avg_metrics': avg_metrics,
                    'per_repeat': noise_results
                }, f, indent=2)

        all_results[model_type] = model_results

    # 保存总结果
    with open(work_dir / 'all_results.json', 'w') as f:
        # 转换为可序列化的格式
        serializable_results = {}
        for model_type, model_res in all_results.items():
            serializable_results[model_type] = {}
            for noise_name, noise_res in model_res.items():
                serializable_results[model_type][noise_name] = {
                    'avg_metrics': noise_res['avg_metrics'],
                    'per_repeat_count': len(noise_res['per_repeat'])
                }
        json.dump(serializable_results, f, indent=2)

    # 生成性能曲线图
    try:
        plot_save_path = work_dir / 'performance_curves.png'
        plot_performance_curves(all_results, str(plot_save_path))
    except Exception as e:
        print(f"生成性能曲线图失败: {e}")

    print(f"\n鲁棒性评估完成！结果保存至: {work_dir}")


if __name__ == '__main__':
    main()
