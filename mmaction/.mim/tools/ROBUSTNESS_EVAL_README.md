# 模型鲁棒性评估实验

## 概述

本工具用于评估模型在不同强度speckle噪声下的鲁棒性性能。实验基于Gamma分布的乘性噪声，测试模型对TRUS图像/视频数据的抗噪能力。

## 主要功能

### 1. Speckle噪声生成
- 基于Gamma分布的乘性噪声：`x -> x * Gamma(k, 1/k)`
- 支持不同噪声强度：
  - Clean：无噪声 (k=∞)
  - Low：低噪声强度 (k=10)
  - Medium：中噪声强度 (k=5)
  - High：高噪声强度 (k=2)

### 2. 性能指标计算
- Accuracy：准确率
- AUC：ROC曲线下面积
- F1-Score：F1分数
- Precision：精确率
- Recall/Sensitivity：召回率/灵敏度
- Specificity：特异性
- Positive/Negative Accuracy：正类/负类准确率

### 3. 模型对比
- Full：完整模型 (含MedSAM + WTCR)
- No-WTCR：不含WTCR分支的baseline
- No-SAM：不含SAM特征的baseline

### 4. 可视化
- 性能随噪声强度变化的曲线图
- 支持直接用于论文发表

## 使用方法

### 方法1：独立运行鲁棒性评估

```bash
cd /data/pqh/env/MM
python tools/robustness_eval.py \
    --config configs/recognition/my_work/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py \
    --checkpoint checkpoints/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-97d0835d.pth \
    --work-dir ./robustness_eval_results \
    --noise-configs clean low medium high \
    --repeats 3 \
    --baseline-types full no_wtcr no_sam
```

### 方法2：在训练脚本中自动启用

```bash
cd /data/pqh/env/MM
python tools/train_pqh.py \
    --config configs/recognition/my_work/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py \
    --robustness_eval \
    --robustness_work_dir ./robustness_eval_results
```

## 参数说明

### 必需参数
- `--config`：模型配置文件路径
- `--checkpoint`：训练好的模型checkpoint路径

### 可选参数
- `--work-dir`：结果保存目录 (默认: ./robustness_eval_results)
- `--noise-configs`：噪声配置列表 (默认: clean low medium high)
- `--repeats`：每个噪声强度重复评估次数 (默认: 3)
- `--baseline-types`：要评估的模型类型 (默认: full no_wtcr no_sam)

## 输出文件结构

```
robustness_eval_results/
├── full/                          # 完整模型结果
│   ├── clean/                     # 无噪声结果
│   │   ├── repeat_0/results.json
│   │   ├── repeat_1/results.json
│   │   └── repeat_2/results.json
│   ├── low/                       # 低噪声结果
│   │   └── ...
│   ├── medium/                    # 中噪声结果
│   │   └── ...
│   └── high/                      # 高噪声结果
│       └── ...
├── no_wtcr/                       # 无WTCR baseline结果
│   └── ...
├── no_sam/                        # 无SAM baseline结果
│   └── ...
├── all_results.json               # 汇总结果
└── performance_curves.png         # 性能曲线图
```

## 结果文件说明

### results.json (各噪声强度结果)
```json
{
  "noise_config": {
    "name": "low",
    "k": 10.0,
    "description": "低噪声强度 (k=10)"
  },
  "avg_metrics": {
    "accuracy": 0.854,
    "auc": 0.912,
    "f1_score": 0.823,
    "precision": 0.867,
    "recall": 0.789,
    "specificity": 0.901,
    "sensitivity": 0.789
  },
  "per_repeat": [...]
}
```

### all_results.json (汇总结果)
```json
{
  "full": {
    "clean": {"avg_metrics": {...}, "per_repeat_count": 3},
    "low": {"avg_metrics": {...}, "per_repeat_count": 3},
    "medium": {"avg_metrics": {...}, "per_repeat_count": 3},
    "high": {"avg_metrics": {...}, "per_repeat_count": 3}
  },
  "no_wtcr": {...},
  "no_sam": {...}
}
```

## 性能曲线图

生成的`performance_curves.png`包含6个子图：
1. Accuracy vs Noise Intensity
2. AUC vs Noise Intensity
3. F1-Score vs Noise Intensity
4. Precision vs Noise Intensity
5. Recall vs Noise Intensity
6. Specificity vs Noise Intensity

每条曲线代表不同模型类型（Full, No-WTCR, No-SAM）的性能变化。

## 技术实现细节

### Speckle噪声模型
- 使用PyTorch的`torch.distributions.Gamma`生成Gamma分布噪声
- 乘性噪声：`noisy = clean * gamma_noise`
- k值越小，噪声方差越大，图像质量越差

### 数据收集
- 通过hook机制收集模型预测结果
- 支持MMEngine的评估流程
- 自动计算多种分类指标

### Baseline模型
- 目前No-WTCR和No-SAM baseline使用占位符实现
- 需要根据具体模型架构进行定制修改

## 注意事项

1. **计算资源**：评估多个噪声强度和重复次数需要较多时间
2. **Baseline实现**：No-WTCR和No-SAM baseline需要根据具体模型实现
3. **噪声仅用于评估**：不影响训练过程，不用于数据增强
4. **Gamma分布参数**：k值可根据需要调整，支持自定义噪声强度

## 依赖项

- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- MMEngine
- MMCV

## 故障排除

### 常见问题
1. **ImportError**: 确保在MM环境中运行，路径设置正确
2. **CUDA内存不足**: 减少batch_size或使用CPU评估
3. **Baseline模型报错**: 检查模型配置文件和checkpoint路径

### 日志调试
运行时会输出详细的进度信息，包括：
- 当前评估的噪声强度
- 重复次数
- 模型类型
- 保存路径
