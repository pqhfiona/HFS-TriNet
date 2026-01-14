import torch
import torch.nn as nn
import torch.nn.functional as F

class WIN3d(nn.Module):
    def __init__(self,
                 num_features: int,
                 window_size: int = 3,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        self.num_features = num_features
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size, window_size)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            # buffers sized by num_features at init
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C, T, H, W]
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (N,C,T,H,W), got {x.dim()}D")

        N, C, T, H, W = x.shape

        # --- 防守式检查：如果模块的 num_features 与输入 C 不匹配，自动修正 buffers（并警告） ---
        if self.track_running_stats and hasattr(self, 'running_mean'):
            if self.running_mean is not None and self.running_mean.numel() != C:
                # 自动重置 running buffers（避免训练中断），同时发出警告
                # 真实场景下建议找到并修正模型定义/替换 BN 的地方，
                # 但为了训练不中断，这里做兼容处理。
                msg = (f"[WIN3d] running_mean length ({self.running_mean.numel()}) != "
                       f"input channels ({C}). Reinitializing running buffers to size {C}. "
                       "Please check that WIN was constructed with correct num_features.")
                # 使用 print 便于在多卡日志中看到
                print(msg)

                device = x.device
                dtype = x.dtype
                # 重新注册 buffers（用新的 tensor 覆盖）
                self.running_mean = torch.zeros(C, device=device, dtype=dtype)
                self.running_var = torch.ones(C, device=device, dtype=dtype)
                self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=device)

                # 若有 affine 参数也要适配（若 affine 与原始 num_features 不同，建议重新init，但此处我们尽量保留）
                if self.affine:
                    # 如果 weight/bias 与 C 不匹配，重新初始化
                    if self.weight is None or self.weight.numel() != C:
                        self.weight = nn.Parameter(torch.ones(C, device=device, dtype=dtype))
                        self.bias = nn.Parameter(torch.zeros(C, device=device, dtype=dtype))

        # 计算局部统计量
        local_mean, local_var = self._compute_local_stats(x)  # shapes: [N,C,T_out,H_out,W_out]

        # 如果 track_running_stats，则更新 running buffers （使用正确的 reduce 维度）
        if self.training or not self.track_running_stats:
            if self.track_running_stats and self.num_batches_tracked is not None:
                # 增计数
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = float(self.momentum)

                # local_mean 的期望应为跨 batch 和空间维度求平均，得到 shape (C,)
                # 对 5D 输入，正确的 reduce 维度为 (0,2,3,4)
                mean_for_running = local_mean.mean(dim=(0, 2, 3, 4))  # -> (C,)
                var_for_running = local_var.mean(dim=(0, 2, 3, 4))    # -> (C,)

                # 确保 running buffers 在同一 device/dtype
                rm = self.running_mean.to(mean_for_running.device, mean_for_running.dtype)
                rv = self.running_var.to(var_for_running.device, var_for_running.dtype)

                # 更新 running stats（广播形状一致）
                self.running_mean = (1 - exponential_average_factor) * rm + exponential_average_factor * mean_for_running
                self.running_var = (1 - exponential_average_factor) * rv + exponential_average_factor * var_for_running
        else:
            # 评估模式下采用 running stats（广播到 N,C,T,H,W）
            local_mean = self.running_mean.view(1, C, 1, 1, 1).expand(N, -1, T, H, W)
            local_var = self.running_var.view(1, C, 1, 1, 1).expand(N, -1, T, H, W)

        # 标准化并应用仿射变换
        x_normalized = (x - local_mean) / torch.sqrt(local_var + self.eps)

        if self.affine:
            x_normalized = x_normalized * self.weight.view(1, C, 1, 1, 1) + \
                           self.bias.view(1, C, 1, 1, 1)

        return x_normalized

    def _compute_local_stats(self, x: torch.Tensor):
        """
        一个内存友好的逐通道（或按 batch）实现，返回 local_mean, local_var
        输入 x shape: [B, C, T, H, W]
        返回 shape: local_mean, local_var -> [B, C, T, H, W]（same spatial dims, no padding shrink）
        说明：这里假设 window_size 为 odd 且我们使用 reflect padding，使输出与输入 T,H,W 对齐。
        """
        B, C, T, H, W = x.shape
        kT, kH, kW = self.window_size

        pad_t = kT // 2
        pad_h = kH // 2
        pad_w = kW // 2
        x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_t, pad_t), mode='reflect')

        # 逐通道卷积，减少一次性 kernel 大小对显存的要求
        local_mean_list = []
        local_var_list = []
        # 为避免在循环内频繁分配 kernel，提前在 cpu 上构造基础 kernel 并移动到 device
        device = x.device
        dtype = x.dtype
        base_kernel = torch.ones(1, 1, kT, kH, kW, device=device, dtype=dtype) / (kT * kH * kW)

        for i in range(C):
            xi = x_padded[:, i:i+1, :, :, :]          # [B,1,T,H,W]
            # 使用同一个 base_kernel 即可（每次都复制会有开销，但 kernel 很小）
            mean_i = F.conv3d(xi, base_kernel, padding=0)            # [B,1,T,H,W]
            sq_i = xi.mul(xi)
            var_i = F.conv3d(sq_i, base_kernel, padding=0) - mean_i * mean_i
            local_mean_list.append(mean_i)
            local_var_list.append(var_i)

        local_mean = torch.cat(local_mean_list, dim=1)   # [B,C,T,H,W]
        local_var = torch.cat(local_var_list, dim=1)     # [B,C,T,H,W]
        return local_mean, local_var
