# WTCR (Wavelet-based Transform Convolutional Reconstruction) 特征域可视化分析工具

## 概述

本工具用于分析WTCR分支对特征域中噪声相关高频成分的抑制效果。该工具实现了两种主要的分析方式：

1. **基于小波变换的子带能量可视化** - 分析LL/LH/HL/HH子带的能量分布
2. **基于FFT的特征频谱幅值可视化** - 分析特征的频率域特性

## 主要功能

### WTCRVisualizer 类

#### 主要方法：

- `visualize_wavelet_subbands()`: 可视化小波子带能量对比
- `visualize_fft_spectrum()`: 可视化FFT频谱对比
- `visualize_feature_comparison()`: 生成完整的特征对比分析
- `visualize_feature_maps()`: 可视化特征激活图

#### 可视化内容：

对同一个样本生成三种特征情况的对比：
1. **原始（clean）特征** - 未添加噪声的干净特征
2. **人工加入speckle噪声后的特征** - 添加乘性speckle噪声
3. **加噪特征经过WTCR处理后的特征** - WTCR分支处理后的结果

## 使用方法

### 1. 独立使用WTCR可视化工具

```python
from wtcr_visualization import WTCRVisualizer

# 创建可视化器
visualizer = WTCRVisualizer(wavelet_type='db1', wt_levels=1)

# 创建WTCR分支
wtcr_branch = visualizer.create_wtcr_branch(in_channels=64, out_channels=64)

# 准备特征数据
clean_feature = torch.randn(2, 64, 32, 32)  # (B, C, H, W)
noisy_feature = visualizer.apply_speckle_noise(clean_feature, var=0.05)

# WTCR处理
wtcr_branch.eval()
with torch.no_grad():
    wtcr_output = wtcr_branch(noisy_feature)

# 生成可视化分析
visualizer.visualize_feature_comparison(
    clean_feature, noisy_feature, wtcr_output,
    save_dir='./analysis_results',
    sample_idx=0,
    noise_var=0.05
)
```

### 2. 在评估脚本中自动启用

修改 `eval_with_speckle.py` 中的配置：

```python
# 启用WTCR可视化分析
enable_wtcr_analysis = True
```

运行评估脚本时会自动生成WTCR分析结果：

```bash
cd /data/pqh/env/MM
conda activate MM
python3 tools/eval_with_speckle.py
```

## 输出文件说明

每次分析会生成三个可视化文件：

### 1. 小波子带能量分析图 (`sample_X_wavelet_energy_var_Y.png`)
- 显示LL（低频-低频）、LH（低频-高频）、HL（高频-低频）、HH（高频-高频）子带的能量对比
- 横轴：子带类型
- 纵轴：能量值
- 三列分别对应：Clean、Noisy、WTCR处理后

### 2. FFT频谱分析图 (`sample_X_fft_spectrum_var_Y.png`)
- 显示特征的二维FFT频谱幅值分布
- 使用对数变换和归一化以便可视化
- 三列分别对应：Clean、Noisy、WTCR处理后

### 3. 特征激活图 (`sample_X_feature_maps_var_Y.png`)
- 显示特征的激活强度（所有通道的L2范数）
- 使用plasma色彩映射
- 三列分别对应：Clean、Noisy、WTCR处理后

## 参数说明

### WTCRVisualizer 初始化参数：

- `wavelet_type` (str): 小波类型，默认为'db1'
- `wt_levels` (int): 小波变换层数，默认为1

### speckle噪声参数：

- `var` (float): 噪声方差，控制噪声强度

## 分析结果解读

### 小波子带能量分析：
- **LL子带**: 代表低频成分，主要包含结构信息
- **LH/HL子带**: 代表水平/垂直方向的高频成分
- **HH子带**: 代表对角线方向的高频成分，噪声通常集中在此

WTCR的有效性可以通过观察：
- 噪声特征中HH子带能量的增加
- WTCR处理后HH子带能量的降低
- LL/LH/HL子带的结构信息保持

### FFT频谱分析：
- 低频区域（中心）通常包含主要结构信息
- 高频区域（边缘）通常包含细节和噪声信息
- WTCR应该抑制高频噪声同时保留低频结构

## 注意事项

1. **特征域分析**: 该工具仅在特征域进行分析，不涉及图像级处理
2. **不影响训练**: 可视化功能不会影响模型的正常训练和推理
3. **speckle噪声**: 仅用于鲁棒性分析，不用于监督去噪训练
4. **计算开销**: 启用WTCR分析会增加少量计算开销，主要用于调试和论文分析

## 文件结构

```
tools/
├── wtcr_visualization.py          # 主要可视化工具类
├── eval_with_speckle.py          # 集成了WTCR分析的评估脚本
└── WTCR_VISUALIZATION_README.md  # 本说明文档
```

## 依赖项

- PyTorch
- NumPy
- Matplotlib
- OpenCV
- PyWavelets (用于小波变换)

确保在MM conda环境中运行，所有依赖已预装。
