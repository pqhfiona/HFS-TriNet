import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import pywt
import cv2
from typing import Tuple, List, Optional, Dict, Any
import math

# 导入WTConv2d等小波变换组件
from TRY.get_feature.WT import WTConv2d, create_wavelet_filter, wavelet_transform


class WTCRVisualizer:
    """
    WTCR (Wavelet-based Transform Convolutional Reconstruction) 分支特征域可视化工具

    用于分析WTCR分支对噪声相关高频成分的抑制效果，可视化方式包括：
    1. 基于小波变换的子带能量可视化（LL/LH/HL/HH）
    2. 基于FFT的特征频谱幅值可视化
    """

    def __init__(self, wavelet_type: str = 'db1', wt_levels: int = 1):
        """
        初始化WTCR可视化器

        Args:
            wavelet_type: 小波类型，默认为'db1'
            wt_levels: 小波变换层数，默认为1
        """
        self.wavelet_type = wavelet_type
        self.wt_levels = wt_levels

        # 创建小波滤波器（用于分析）
        self.dec_filters, self.rec_filters = create_wavelet_filter(
            wavelet_type, in_size=1, out_size=1, type=torch.float32
        )

    def apply_speckle_noise(self, feature: torch.Tensor, var: float) -> torch.Tensor:
        """
        对特征应用乘性speckle噪声

        Args:
            feature: 输入特征张量 (C, H, W) 或 (B, C, H, W) 或 (B, C, T, H, W)
            var: 噪声方差

        Returns:
            加噪后的特征
        """
        if var == 0:
            return feature.clone()

        noise = torch.randn_like(feature) * math.sqrt(var)
        noisy_feature = feature + feature * noise
        return noisy_feature

    def create_wtcr_branch(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        创建WTCR分支模型

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数

        Returns:
            WTCR分支模型
        """
        return WTConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            wt_levels=self.wt_levels,
            wt_type=self.wavelet_type
        )

    def apply_wtcr_to_video_feature(self, feature: torch.Tensor, wtcr_branch: nn.Module) -> torch.Tensor:
        """
        对视频特征应用WTCR分支

        Args:
            feature: 输入特征 (B, C, T, H, W)
            wtcr_branch: WTCR分支模型

        Returns:
            WTCR处理后的特征 (B, C, T, H, W)
        """
        with torch.no_grad():
            # 将 (B, C, T, H, W) 重塑为 (B*T, C, H, W) 来适应WTConv2d
            B, C, T, H, W = feature.shape
            feature_reshaped = feature.view(B * T, C, H, W)

            wtcr_output_reshaped = wtcr_branch(feature_reshaped)

            # 恢复原始形状 (B, C, T, H, W)
            wtcr_output = wtcr_output_reshaped.view(B, C, T, H, W)

        return wtcr_output

    def extract_wavelet_subbands(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        从特征中提取小波子带

        Args:
            feature: 输入特征 (C, H, W) 或 (B, C, H, W)

        Returns:
            包含LL, LH, HL, HH子带的字典
        """
        # 处理输入格式
        if feature.dim() == 3:  # (C, H, W)
            feature = feature.unsqueeze(0)  # (1, C, H, W)
        elif feature.dim() == 5:  # (B, C, T, H, W)
            # 取第一帧进行分析
            feature = feature[:, :, 0, :, :]  # (B, C, H, W)
        elif feature.dim() != 4:
            raise ValueError(f"Unsupported feature dimensions: {feature.shape}")

        B, C, H, W = feature.shape

        # 计算每个通道的小波变换
        subbands = {'LL': [], 'LH': [], 'HL': [], 'HH': []}

        for c in range(C):
            channel_feat = feature[:, c:c+1, :, :]  # (B, 1, H, W)

            # 应用小波变换
            wt_result = wavelet_transform(channel_feat, self.dec_filters.to(feature.device))
            # wt_result shape: (B, 1, 4, H//2, W//2)

            # 分离子带
            ll = wt_result[:, :, 0, :, :]  # LL: (B, 1, H//2, W//2)
            lh = wt_result[:, :, 1, :, :]  # LH: (B, 1, H//2, W//2)
            hl = wt_result[:, :, 2, :, :]  # HL: (B, 1, H//2, W//2)
            hh = wt_result[:, :, 3, :, :]  # HH: (B, 1, H//2, W//2)

            subbands['LL'].append(ll)
            subbands['LH'].append(lh)
            subbands['HL'].append(hl)
            subbands['HH'].append(hh)

        # 堆叠所有通道
        for key in subbands:
            subbands[key] = torch.cat(subbands[key], dim=1)  # (B, C, H//2, W//2)

        return subbands

    def compute_wavelet_energy(self, subbands: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算各子带的能量

        Args:
            subbands: 小波子带字典

        Returns:
            各子带能量
        """
        energies = {}
        for key, subband in subbands.items():
            # 计算能量：平方和
            energy = torch.sum(subband ** 2, dim=[1, 2, 3], keepdim=True)  # (B, 1, 1, 1)
            energies[key] = energy
        return energies

    def compute_fft_spectrum(self, feature: torch.Tensor) -> torch.Tensor:
        """
        计算特征的FFT频谱幅值

        Args:
            feature: 输入特征 (C, H, W) 或 (B, C, H, W)

        Returns:
            FFT频谱幅值 (B, C, H, W)
        """
        # 处理输入格式
        if feature.dim() == 3:  # (C, H, W)
            feature = feature.unsqueeze(0)  # (1, C, H, W)
        elif feature.dim() == 5:  # (B, C, T, H, W)
            # 取第一帧进行分析
            feature = feature[:, :, 0, :, :]  # (B, C, H, W)

        B, C, H, W = feature.shape

        # 对每个通道进行FFT
        spectra = []
        for c in range(C):
            channel_feat = feature[:, c, :, :]  # (B, H, W)

            # 2D FFT
            fft_result = torch.fft.fft2(channel_feat)
            spectrum = torch.abs(fft_result)  # 幅值

            # 移到中心
            spectrum = torch.fft.fftshift(spectrum, dim=[-2, -1])

            spectra.append(spectrum)

        # 堆叠通道
        spectrum_tensor = torch.stack(spectra, dim=1)  # (B, C, H, W)

        return spectrum_tensor

    def visualize_wavelet_subbands(self,
                                   clean_feature: torch.Tensor,
                                   noisy_feature: torch.Tensor,
                                   wtcr_output: torch.Tensor,
                                   save_path: str,
                                   title: str = "Wavelet Subband Energy Analysis") -> None:
        """
        可视化小波子带能量对比

        Args:
            clean_feature: 原始特征
            noisy_feature: 加噪特征
            wtcr_output: WTCR处理后的特征
            save_path: 保存路径
            title: 图表标题
        """
        # 提取子带
        clean_subbands = self.extract_wavelet_subbands(clean_feature)
        noisy_subbands = self.extract_wavelet_subbands(noisy_feature)
        wtcr_subbands = self.extract_wavelet_subbands(wtcr_output)

        # 计算能量
        clean_energies = self.compute_wavelet_energy(clean_subbands)
        noisy_energies = self.compute_wavelet_energy(noisy_subbands)
        wtcr_energies = self.compute_wavelet_energy(wtcr_subbands)

        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=14)

        # 子带名称
        subband_names = ['LL (Low-Low)', 'LH (Low-High)', 'HL (High-Low)', 'HH (High-High)']
        subband_keys = ['LL', 'LH', 'HL', 'HH']

        # 数据
        conditions = ['Clean', 'Noisy', 'WTCR']
        energies_data = [clean_energies, noisy_energies, wtcr_energies]

        x = np.arange(len(subband_keys))
        width = 0.25

        for i, (cond, energies) in enumerate(zip(conditions, energies_data)):
            energy_values = [energies[key].mean().item() for key in subband_keys]
            axes[i].bar(x + i*width, energy_values, width, label=cond, alpha=0.7)
            axes[i].set_title(f'{cond} Feature Subband Energies')
            axes[i].set_xticks(x + width)
            axes[i].set_xticklabels(subband_names, rotation=45)
            axes[i].set_ylabel('Energy')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_fft_spectrum(self,
                              clean_feature: torch.Tensor,
                              noisy_feature: torch.Tensor,
                              wtcr_output: torch.Tensor,
                              save_path: str,
                              title: str = "FFT Spectrum Analysis") -> None:
        """
        可视化FFT频谱对比

        Args:
            clean_feature: 原始特征
            noisy_feature: 加噪特征
            wtcr_output: WTCR处理后的特征
            save_path: 保存路径
            title: 图表标题
        """
        # 计算频谱
        clean_spectrum = self.compute_fft_spectrum(clean_feature)
        noisy_spectrum = self.compute_fft_spectrum(noisy_feature)
        wtcr_spectrum = self.compute_fft_spectrum(wtcr_output)

        # 取第一个样本和第一个通道进行可视化
        clean_spec = clean_spectrum[0, 0].cpu().numpy()
        noisy_spec = noisy_spectrum[0, 0].cpu().numpy()
        wtcr_spec = wtcr_spectrum[0, 0].cpu().numpy()

        # 对数变换以便可视化
        clean_spec_log = np.log1p(clean_spec)
        noisy_spec_log = np.log1p(noisy_spec)
        wtcr_spec_log = np.log1p(wtcr_spec)

        # 归一化到0-1
        def normalize(arr):
            arr_min, arr_max = arr.min(), arr.max()
            return (arr - arr_min) / (arr_max - arr_min + 1e-8)

        clean_spec_norm = normalize(clean_spec_log)
        noisy_spec_norm = normalize(noisy_spec_log)
        wtcr_spec_norm = normalize(wtcr_spec_log)

        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=14)

        # 显示频谱
        im1 = axes[0].imshow(clean_spec_norm, cmap='viridis', aspect='equal')
        axes[0].set_title('Clean Feature FFT Spectrum')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        im2 = axes[1].imshow(noisy_spec_norm, cmap='viridis', aspect='equal')
        axes[1].set_title('Noisy Feature FFT Spectrum')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        im3 = axes[2].imshow(wtcr_spec_norm, cmap='viridis', aspect='equal')
        axes[2].set_title('WTCR Processed FFT Spectrum')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)

        # 设置坐标标签
        for ax in axes:
            ax.set_xlabel('Frequency (W)')
            ax.set_ylabel('Frequency (H)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_feature_comparison(self,
                                   clean_feature: torch.Tensor,
                                   noisy_feature: torch.Tensor,
                                   wtcr_output: torch.Tensor,
                                   save_dir: str,
                                   sample_idx: int = 0,
                                   noise_var: float = 0.05,
                                   wtcr_branch: Optional[nn.Module] = None) -> None:
        """
        生成完整的特征对比可视化

        Args:
            clean_feature: 原始特征 (B, C, H, W) 或 (B, C, T, H, W)
            noisy_feature: 加噪特征
            wtcr_output: WTCR处理后的特征
            save_dir: 保存目录
            sample_idx: 样本索引
            noise_var: 噪声方差
        """
        os.makedirs(save_dir, exist_ok=True)

        # 小波子带能量可视化
        wavelet_path = os.path.join(save_dir, f'sample_{sample_idx}_wavelet_energy_var_{noise_var:.3f}.png')
        self.visualize_wavelet_subbands(
            clean_feature, noisy_feature, wtcr_output, wavelet_path,
            f"Sample {sample_idx} - Wavelet Subband Energy Analysis (Noise Var: {noise_var})"
        )

        # FFT频谱可视化
        fft_path = os.path.join(save_dir, f'sample_{sample_idx}_fft_spectrum_var_{noise_var:.3f}.png')
        self.visualize_fft_spectrum(
            clean_feature, noisy_feature, wtcr_output, fft_path,
            f"Sample {sample_idx} - FFT Spectrum Analysis (Noise Var: {noise_var})"
        )

        # 特征激活图可视化（可选）
        self.visualize_feature_maps(clean_feature, noisy_feature, wtcr_output,
                                   os.path.join(save_dir, f'sample_{sample_idx}_feature_maps_var_{noise_var:.3f}.png'),
                                   sample_idx, noise_var)

    def visualize_feature_maps(self,
                              clean_feature: torch.Tensor,
                              noisy_feature: torch.Tensor,
                              wtcr_output: torch.Tensor,
                              save_path: str,
                              sample_idx: int = 0,
                              noise_var: float = 0.05) -> None:
        """
        可视化特征激活图

        Args:
            clean_feature: 原始特征
            noisy_feature: 加噪特征
            wtcr_output: WTCR处理后的特征
            save_path: 保存路径
            sample_idx: 样本索引
            noise_var: 噪声方差
        """
        # 处理输入格式
        if clean_feature.dim() == 5:  # (B, C, T, H, W)
            clean_feature = clean_feature[:, :, 0, :, :]  # 取第一帧
            noisy_feature = noisy_feature[:, :, 0, :, :]
            wtcr_output = wtcr_output[:, :, 0, :, :]

        # 取第一个样本
        clean_feat = clean_feature[0]  # (C, H, W)
        noisy_feat = noisy_feature[0]
        wtcr_feat = wtcr_output[0]

        # 计算每个通道的激活强度（L2范数）
        clean_activation = torch.norm(clean_feat, dim=0)  # (H, W)
        noisy_activation = torch.norm(noisy_feat, dim=0)
        wtcr_activation = torch.norm(wtcr_feat, dim=0)

        # 转换为numpy并归一化
        def normalize(arr):
            arr = arr.cpu().numpy()
            arr_min, arr_max = arr.min(), arr.max()
            return (arr - arr_min) / (arr_max - arr_min + 1e-8)

        clean_act_norm = normalize(clean_activation)
        noisy_act_norm = normalize(noisy_activation)
        wtcr_act_norm = normalize(wtcr_activation)

        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Sample {sample_idx} - Feature Activation Maps (Noise Var: {noise_var})', fontsize=14)

        # 显示激活图
        im1 = axes[0].imshow(clean_act_norm, cmap='plasma', aspect='equal')
        axes[0].set_title('Clean Feature Activation')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        im2 = axes[1].imshow(noisy_act_norm, cmap='plasma', aspect='equal')
        axes[1].set_title('Noisy Feature Activation')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        im3 = axes[2].imshow(wtcr_act_norm, cmap='plasma', aspect='equal')
        axes[2].set_title('WTCR Processed Activation')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_wtcr_analysis_demo():
    """
    创建WTCR分析演示
    """
    # 创建可视化器
    visualizer = WTCRVisualizer(wavelet_type='db1', wt_levels=1)

    # 创建WTCR分支
    wtcr_branch = visualizer.create_wtcr_branch(in_channels=64, out_channels=64)

    # 生成测试特征
    batch_size, channels, height, width = 2, 64, 32, 32
    clean_feature = torch.randn(batch_size, channels, height, width)

    # 添加speckle噪声
    noise_var = 0.05
    noisy_feature = visualizer.apply_speckle_noise(clean_feature, noise_var)

    # 通过WTCR分支处理
    wtcr_branch.eval()
    with torch.no_grad():
        wtcr_output = wtcr_branch(noisy_feature)

    # 生成可视化
    save_dir = './wtcr_analysis_demo'
    visualizer.visualize_feature_comparison(
        clean_feature, noisy_feature, wtcr_output,
        save_dir, sample_idx=0, noise_var=noise_var
    )

    print(f"WTCR analysis visualization saved to: {save_dir}")


if __name__ == '__main__':
    create_wtcr_analysis_demo()
