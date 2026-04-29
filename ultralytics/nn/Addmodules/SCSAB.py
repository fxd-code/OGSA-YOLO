import math  # 引入math模块，用于log计算

import torch
import torch.nn as nn
import torch.nn.functional as F  # 引入F模块，用于 interpolate
from einops import rearrange
from mmengine.model import BaseModule

# 定义模块的对外接口
__all__ = ["C2PSASCSAB", "SCSAB", "Conv", "PSABlock"]


class SCSAB(BaseModule):
    """FULLY UPGRADED SCSAB Module (Directional 2D Enhancement + Hybrid Channel Attention). This version includes: 1.
    Direction-aware strip convolutions for enhanced edge detection and directional feature perception (IMPROVEMENT
    1). 2. Hybrid channel attention (Self-Attention + Convolutional Attention) with adaptive fusion and refined
    self-attention channel weight generation for robust large object and offset detection (IMPROVEMENT 2).
    """

    def __init__(
        self,
        dim: int,
        head_num: int = 4,
        window_size: int = 7,  # Downsampling window size for self-attention branch
        group_kernel_sizes: list[int] = [3, 5, 7, 9],  # Kernel sizes for SMSA's 1D DWCs
        qkv_bias: bool = False,
        fuse_bn: bool = False,  # Unused in this version
        norm_cfg: dict = dict(type="BN"),  # Unused directly, GroupNorm is used
        act_cfg: dict = dict(type="ReLU"),  # Unused directly, Sigmoid/Softmax is used
        down_sample_mode: str = "avg_pool",  # Downsampling mode for self-attention branch
        attn_drop_ratio: float = 0.0,
        gate_layer: str = "sigmoid",  # Activation for attention gates
    ):
        super().__init__()
        self.dim = dim

        # Head configuration for self-attention
        _head_num = dim // 64
        if _head_num == 0:
            _head_num = 1
        self.head_num = _head_num  # Ensure correct attribute assignment
        self.head_dim = dim // self.head_num
        self.scaler = self.head_dim**-0.5

        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.down_sample_mode = down_sample_mode

        assert self.dim % 4 == 0, "The dimension of input feature should be divisible by 4."
        self.group_chans = group_chans = self.dim // 4

        # === [IMPROVEMENT 1] Directional 2D Enhancement (Pre-SMSA) ===
        strip_kernel_size = 7  # 可调参数，影响方向性感知范围
        self.strip_conv_h = nn.Conv2d(
            dim, dim, kernel_size=(1, strip_kernel_size), padding=(0, strip_kernel_size // 2), groups=dim, bias=False
        )
        self.strip_conv_w = nn.Conv2d(
            dim, dim, kernel_size=(strip_kernel_size, 1), padding=(strip_kernel_size // 2, 0), groups=dim, bias=False
        )
        self.bn_h_pre = nn.BatchNorm2d(dim)  # Batch norm for pre-enhancement
        self.bn_w_pre = nn.BatchNorm2d(dim)

        # Original SMSA components
        self.local_dwc = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[0],
            padding=group_kernel_sizes[0] // 2,
            groups=group_chans,
        )
        self.global_dwc_s = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[1],
            padding=group_kernel_sizes[1] // 2,
            groups=group_chans,
        )
        self.global_dwc_m = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[2],
            padding=group_kernel_sizes[2] // 2,
            groups=group_chans,
        )
        self.global_dwc_l = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[3],
            padding=group_kernel_sizes[3] // 2,
            groups=group_chans,
        )
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == "softmax" else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)  # Group norm for H-attention
        self.norm_w = nn.GroupNorm(4, dim)  # Group norm for W-attention

        # Original PCSA components (now integrated into Hybrid Channel Attention)
        self.conv_d = nn.Identity()  # Placeholder for potential future dim reduction
        self.norm_p_sa = nn.GroupNorm(1, dim)  # Group norm for PCSA self-attention (equiv. to LayerNorm)
        # Q, K, V projections are depthwise convolutions, as per original SCSA paper's intent
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)

        # Downsampling for self-attention branch
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        elif down_sample_mode == "avg_pool":
            self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
        elif down_sample_mode == "max_pool":
            self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)
        # 'recombination' mode from original context is not implemented here for simplicity,
        # defaulting to avg_pool if window_size is not -1.

        # === [IMPROVEMENT 2] Hybrid Channel Attention components ===
        # 1. Refined Self-Attention Channel Weight Generation (sub-improvement of 2)
        # This 1x1 conv learns how to best summarize spatial info into a channel-wise weight
        self.sa_summary_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # 2. Convolutional Attention (ECA-like) components (Local Dependencies)
        k_size = int(abs((math.log(dim, 2) + 1) / 2))  # Dynamic kernel size calculation
        k_size = k_size if k_size % 2 else k_size + 1  # Ensure odd kernel size
        self.conv_ca = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        # 3. Adaptive Fusion Gate (learns to combine SA and CA weights)
        self.fusion_gate_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),  # Concatenate both weights and process with 1x1 conv
            nn.Sigmoid(),  # Generate gate values (0-1)
        )
        self.final_ca_gate = (
            nn.Softmax(dim=1) if gate_layer == "softmax" else nn.Sigmoid()
        )  # Final activation for combined weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h_orig, w_orig = x.size()  # Store original H, W for potential interpolation

        # === [IMPROVEMENT 1] Directional 2D Enhancement ===
        # Apply strip convolutions to capture directional features (e.g., edges of coal, belt offset)
        x_h_enhanced = self.bn_h_pre(self.strip_conv_h(x))
        x_w_enhanced = self.bn_w_pre(self.strip_conv_w(x))
        x_enhanced = x + x_h_enhanced + x_w_enhanced  # Residual connection to original feature

        # === 1. Spatial attention priority calculation (SMSA part) ===
        # SMSA operates on the enhanced feature x_enhanced
        x_h = x_enhanced.mean(dim=3)  # Average along width to get H-axis feature (B, C, H)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)  # Split into 4 groups

        x_w = x_enhanced.mean(dim=2)  # Average along height to get W-axis feature (B, C, W)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)  # Split into 4 groups

        # Apply multi-scale 1D depthwise convolutions and concatenate
        x_h_attn = self.sa_gate(
            self.norm_h(
                torch.cat(
                    (
                        self.local_dwc(l_x_h),
                        self.global_dwc_s(g_x_h_s),
                        self.global_dwc_m(g_x_h_m),
                        self.global_dwc_l(g_x_h_l),
                    ),
                    dim=1,
                )
            )
        )
        x_h_attn = x_h_attn.view(b, c, h_orig, 1)  # Reshape to (B, C, H, 1)

        x_w_attn = self.sa_gate(
            self.norm_w(
                torch.cat(
                    (
                        self.local_dwc(l_x_w),
                        self.global_dwc_s(g_x_w_s),
                        self.global_dwc_m(g_x_w_m),
                        self.global_dwc_l(g_x_w_l),
                    ),
                    dim=1,
                )
            )
        )
        x_w_attn = x_w_attn.view(b, c, 1, w_orig)  # Reshape to (B, C, 1, W)

        # Apply spatial attention to the enhanced feature
        x_smsa_out = x_enhanced * x_h_attn * x_w_attn

        # === 2. [IMPROVEMENT 2] Hybrid Channel Attention Calculation ===
        # --- Branch 1: Self-Attention (Global Channel Dependencies) ---
        # Reduce calculations by downsampling
        y_sa = self.down_func(x_smsa_out)
        y_sa = self.conv_d(y_sa)  # conv_d is Identity by default
        _, _, h_s, w_s = y_sa.size()  # H_s, W_s are downsampled dimensions

        y_sa = self.norm_p_sa(y_sa)  # Apply norm for PCSA self-attention
        q = self.q(y_sa)  # Generate Query (depthwise conv)
        k = self.k(y_sa)  # Generate Key (depthwise conv)
        v = self.v(y_sa)  # Generate Value (depthwise conv)

        # Rearrange for multi-head self-attention calculation in channel dimension
        # (B, C, H_s, W_s) -> (B, head_num, head_dim, N_s) where N_s = H_s * W_s
        q = rearrange(
            q, "b (head_num head_dim) h w -> b head_num head_dim (h w)", head_num=self.head_num, head_dim=self.head_dim
        )
        k = rearrange(
            k, "b (head_num head_dim) h w -> b head_num head_dim (h w)", head_num=self.head_num, head_dim=self.head_dim
        )
        v = rearrange(
            v, "b (head_num head_dim) h w -> b head_num head_dim (h w)", head_num=self.head_num, head_dim=self.head_dim
        )

        # Self-attention calculation: (Q @ K_T) * scaler @ V
        attn_sa_raw = q @ k.transpose(-2, -1) * self.scaler
        attn_sa_raw = self.attn_drop(attn_sa_raw.softmax(dim=-1))
        attn_sa_raw = attn_sa_raw @ v  # (B, head_num, head_dim, N_s)

        # Rearrange back to (B, C, H_s, W_s)
        attn_sa_rearranged = rearrange(
            attn_sa_raw, "b head_num head_dim (h w) -> b (head_num head_dim) h w", h=h_s, w=w_s
        )

        # Refined Self-Attention Channel Weight Generation:
        # Interpolate back to original resolution if downsampled, then summarize using a 1x1 conv + spatial mean
        if h_s != h_orig or w_s != w_orig:
            attn_sa_full_res = F.interpolate(
                attn_sa_rearranged, size=(h_orig, w_orig), mode="bilinear", align_corners=False
            )
        else:  # No downsampling occurred
            attn_sa_full_res = attn_sa_rearranged

        attn_sa_weights = self.sa_summary_conv(attn_sa_full_res).mean((2, 3), keepdim=True)  # (B, C, 1, 1)

        # --- Branch 2: Convolutional Attention (Local Channel Dependencies - ECA-like) ---
        # Global average pool to get channel descriptor (B, C, 1, 1), then reshape to (B, 1, C) for 1D conv
        y_conv = F.adaptive_avg_pool2d(x_smsa_out, 1).squeeze(-1).transpose(-1, -2)
        attn_conv_weights = (
            self.conv_ca(y_conv).transpose(-1, -2).unsqueeze(-1)
        )  # Apply 1D conv, reshape to (B, C, 1, 1)

        # --- Adaptive Fusion of Hybrid Channel Attention Weights ---
        # Concatenate weights from both branches (B, 2*C, 1, 1)
        fused_attn_inputs = torch.cat([attn_sa_weights, attn_conv_weights], dim=1)
        # Generate adaptive gate values (B, C, 1, 1)
        gate = self.fusion_gate_conv(fused_attn_inputs)

        # Adaptive weighted sum: gate * SA_weight + (1 - gate) * Conv_weight
        combined_attn_weights = attn_sa_weights * gate + attn_conv_weights * (1 - gate)
        # Apply final activation (Sigmoid/Softmax)
        final_attn_map = self.final_ca_gate(combined_attn_weights)

        # === Apply Final Channel Attention (Residual-like application) ===
        # This applies the combined channel attention map to the SMSA output
        return x_smsa_out + x_smsa_out * final_attn_map


# --- The following classes are kept for yaml file compatibility ---
# (They are unchanged from your original file)
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks. This class encapsulates the
    functionality for applying multi-head attention and feed-forward neural network layers with optional
    shortcut connections.
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = SCSAB(c)  # PSABlock now uses the improved SCSAB
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSASCSAB(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing. This module implements a
    convolutional block with attention mechanisms to enhance feature extraction and processing capabilities. It
    includes a series of PSABlock modules for self-attention and feed-forward operations.
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


#    当前SCSAB模块（你提供的代码）的功能概述
# 在详细分析YAML文件之前，我们先快速回顾一下你这个版本的SCSAB模块所具备的关键功能：
# [改进1] 方向性2D增强 (Directional 2D Enhancement)：
# 在进入核心注意力计算之前，通过水平和垂直的深度可分离条状卷积对输入特征进行预增强。
# 目的：更早、更直接地捕获图像中的边缘信息和方向性特征。这对于左偏移的识别（方向性）和大块煤的轮廓分割（边缘）至关重要。
# SMSA (共享多语义空间注意力)：
# 沿H和W维度进行平均池化。
# 将通道分组，并对每个组应用多尺度（3, 5, 7, 9）的1D深度可分离卷积，以捕捉不同尺度的空间上下文。
# 通过GroupNorm和Sigmoid/Softmax生成空间注意力权重。
# 目的：提供多尺度的空间先验，对图像区域的重要性进行初步加权。
# [改进2] 混合通道注意力 (Hybrid Channel Attention)：
# 自注意力分支 (Global Dependencies)：
# 对SMSA输出进行下采样（down_func）。
# QKV投影采用深度可分离的1x1 Conv2d (参数量 3 * dim)，确保轻量级。
# 计算多头自注意力，捕获通道间的全局、长距离依赖。
# 精炼通道权重生成：自注意力输出上采样回原分辨率后，通过一个1x1 Conv2d (sa_summary_conv)，学习如何从复杂的空间-通道特征图中提炼出更有效的通道注意力权重，然后才进行空间平均。
# 目的：更智能地总结自注意力捕获的全局上下文，以获得高代表性的通道权重，这对于理解大块煤的整体语义和复杂偏移模式有益。
# 卷积注意力分支 (Local Dependencies)：
# 对SMSA输出进行全局平均池化。
# 应用一个动态核大小的1D卷积 (conv_ca)，捕获相邻通道间的局部交互。
# 目的：高效地捕捉局部通道依赖，对于理解煤块的纹理细节和皮带材质特性有帮助。
# 自适应融合 (Adaptive Fusion)：
# 通过一个可学习的门控机制 (fusion_gate_conv)，根据输入特征动态调整自注意力权重和卷积注意力权重的融合比例。
# 目的：让模型灵活地权衡全局（自注意力）和局部（卷积注意力）通道信息的重要性，使得通道注意力更加鲁棒和强大。
# 最终应用：将融合后的通道注意力图以残差连接的方式 (x_smsa_out + x_smsa_out * final_attn_map) 作用于SMSA的输出。
# 目的：在不损失原始信息的前提下，对特征进行通道维度的增强和调整，促进信息流动和性能提升。
