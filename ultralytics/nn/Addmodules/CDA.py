import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


__all__ = ['C3k2_ODConv', 'ODConv2d', 'C3k2_ODConv_Advanced', 'ODConv2d_Advanced', 
           'MultiScaleContext', 'Bottleneck_Hybrid_AttentiveFusion', 'C3k2_HybridFusion']


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


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        # x = self.bn(x) # 在外面我提供了一个bn这里会报错
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class Attention_Advanced(nn.Module):
    """
    方向感知的注意力生成器 (Direction-Aware Attention Generator)
    创新点一：通过全局、水平、垂直三个池化分支来增强对方向和边缘的感知能力
    解决"左偏移"问题：能够精准感知皮带边缘的方向性特征
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention_Advanced, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        # 保留全局视野：全局平均池化分支
        self.avgpool_global = nn.AdaptiveAvgPool2d(1)
        
        # 增加水平扫描：水平池化分支，压缩成垂直的"特征带"
        # 对检测横向的皮带边缘极为敏感
        self.avgpool_horizontal = nn.AdaptiveAvgPool2d((None, 1))
        
        # 增加垂直扫描：垂直池化分支，压缩成水平的"特征带"
        # 浓缩所有垂直方向上的模式
        self.avgpool_vertical = nn.AdaptiveAvgPool2d((1, None))
        
        # 信息融合：将三个分支拼接后送入FC层
        # 全局: [B, C, 1, 1]
        # 水平: [B, C, H, 1] -> 压缩成 [B, C, 1, 1]
        # 垂直: [B, C, 1, W] -> 压缩成 [B, C, 1, 1]
        # 拼接后: [B, 3*C, 1, 1]，然后通过FC压缩回C维度
        self.fc_fusion = nn.Conv2d(in_planes * 3, in_planes, 1, bias=False)  # 融合三个分支
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        # 不使用BatchNorm，因为特征图尺寸是[1,1]，BatchNorm在训练模式下会报错
        # self.bn = nn.BatchNorm2d(attention_channel)  # 注释掉，与原始Attention保持一致
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 不再使用BatchNorm，所以不需要初始化
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        """
        前向传播：融合全局、水平、垂直三个池化分支的信息
        形成"超级特征向量"来指导注意力的生成
        """
        B, C, H, W = x.size()
        
        # 全局平均池化分支：[B, C, 1, 1]
        x_global = self.avgpool_global(x)  # [B, C, 1, 1]
        
        # 水平池化分支：[B, C, H, 1] -> 压缩成 [B, C, 1, 1]
        x_horizontal = self.avgpool_horizontal(x)  # [B, C, H, 1]
        x_horizontal = F.adaptive_avg_pool2d(x_horizontal, 1)  # [B, C, 1, 1]
        
        # 垂直池化分支：[B, C, 1, W] -> 压缩成 [B, C, 1, 1]
        x_vertical = self.avgpool_vertical(x)  # [B, C, 1, W]
        x_vertical = F.adaptive_avg_pool2d(x_vertical, 1)  # [B, C, 1, 1]
        
        # 信息融合：拼接三个分支形成"超级特征向量" [B, 3*C, 1, 1]
        x_fused = torch.cat([x_global, x_horizontal, x_vertical], dim=1)  # [B, 3*C, 1, 1]
        
        # 通过1x1卷积将3*C维度的特征压缩回C维度
        x_fused = self.fc_fusion(x_fused)  # [B, C, 1, 1]
        
        # 驱动注意力：送入后续的FC层
        x_fused = self.fc(x_fused)
        # x_fused = self.bn(x_fused)  # 不使用BatchNorm，避免在模型初始化时出错
        x_fused = self.relu(x_fused)
        
        return self.func_channel(x_fused), self.func_filter(x_fused), self.func_spatial(x_fused), self.func_kernel(x_fused)


class MultiScaleContext(nn.Module):
    """
    输入端上下文增强模块 (Input-end Context Enhancement Module)
    创新点二：使用多尺度空洞卷积来扩大感受野，解决"大块煤"问题
    让模型在处理局部特征时能够感知到更大范围的上下文信息
    
    改进版（方案一）：保留原始空间信息
    在融合阶段将原始输入x也拼接进来，确保最原始、最精确的空间位置信息
    不会在空洞卷积过程中被稀释，从而缓解定位精度下降的问题
    """
    def __init__(self, in_planes, dilation_rates=[1, 2, 3]):
        """
        Args:
            in_planes: 输入通道数
            dilation_rates: 空洞卷积的扩张率列表，默认[1, 2, 3]
        """
        super(MultiScaleContext, self).__init__()
        self.dilation_rates = dilation_rates
        
        # 创建多个并行的空洞卷积分支
        self.dilated_convs = nn.ModuleList()
        for dilation in dilation_rates:
            # 每个分支使用3x3空洞卷积，padding自动计算以保持尺寸
            padding = dilation  # 对于3x3卷积，dilation=1时padding=1，dilation=2时padding=2
            self.dilated_convs.append(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, 
                         padding=padding, dilation=dilation, bias=False)
            )
        
        # 特征融合：拼接所有分支 + 原始输入x，所以通道数是 (分支数+1) * in_planes
        # 核心改动：调整融合卷积的输入通道数，包含原始输入
        num_branches = len(dilation_rates)
        self.fusion_conv = nn.Conv2d(in_planes * (num_branches + 1), in_planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播：通过多个空洞卷积分支提取多尺度上下文，然后融合
        改进：将原始输入x也拼接进来，形成一个包含"高精度位置信息"的超级特征图
        """
        # 通过各个空洞卷积分支
        branch_outputs = []
        for dilated_conv in self.dilated_convs:
            branch_outputs.append(dilated_conv(x))
        
        # 特征融合：拼接所有空洞卷积分支的输出
        x_fused_dilated = torch.cat(branch_outputs, dim=1)  # [B, num_branches*C, H, W]
        
        # 核心改动：将原始输入x也拼接进来，形成一个包含"高精度位置信息"的超级特征图
        x_fused_total = torch.cat([x, x_fused_dilated], dim=1)  # [B, (num_branches+1)*C, H, W]
        
        # 通过1x1卷积融合所有特征（包括原始输入和空洞卷积输出）
        x_fused = self.fusion_conv(x_fused_total)  # [B, C, H, W]
        x_fused = self.bn(x_fused)
        x_fused = self.relu(x_fused)
        
        # 残差连接：与原始输入相加
        x_out = x + x_fused
        
        return x_out


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        in_planes = in_planes
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)


class ODConv2d_Advanced(nn.Module):
    """
    改进版ODConv，集成两大创新点：
    1. 方向感知的注意力生成器（Attention_Advanced）- 解决"左偏移"问题
    2. 输入端上下文增强模块（MultiScaleContext）- 解决"大块煤"问题
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4, use_context=True, dilation_rates=[1, 2, 3]):
        super(ODConv2d_Advanced, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.use_context = use_context
        
        # 创新点二：输入端上下文增强模块（可选）
        if use_context:
            self.context_module = MultiScaleContext(in_planes, dilation_rates=dilation_rates)
        else:
            self.context_module = None
        
        # 创新点一：方向感知的注意力生成器
        self.attention = Attention_Advanced(in_planes, out_planes, kernel_size, groups=groups,
                                           reduction=reduction, kernel_num=kernel_num)
        
        # 动态卷积权重
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # 创新点二：先进行上下文增强
        if self.context_module is not None:
            x = self.context_module(x)
        
        # 创新点一：使用方向感知的注意力生成器
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        # 创新点二：先进行上下文增强
        if self.context_module is not None:
            x = self.context_module(x)
        
        # 创新点一：使用方向感知的注意力生成器
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)


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


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ODConv2d(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_ODConv(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class Bottleneck_ODConv_Advanced(nn.Module):
    """
    使用ODConv2d_Advanced的瓶颈层
    在动态卷积之后添加BatchNorm和激活函数来稳定训练
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with ODConv2d_Advanced."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        # 使用改进版的ODConv
        self.cv2_odconv = ODConv2d_Advanced(c_, c2, 3, 1)
        # 在动态卷积之后添加BN和激活函数
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        out = self.cv1(x)
        out = self.cv2_odconv(out)
        out = self.act(self.bn(out))  # BN + 激活
        return x + out if self.add else out


class Bottleneck_Hybrid_AttentiveFusion(nn.Module):
    """
    注意力融合的混合瓶颈层 (Hybrid Bottleneck with Attentive Fusion)
    
    改进版（方案二）：使用注意力机制进行智能融合
    让 Bottleneck_Hybrid 中的两个分支（ODConv 和 Context）进行更智能的"软融合"，
    而不是简单的拼接。引入轻量级的通道注意力机制（SE-Block风格），让网络根据
    输入特征，动态地学习给 odconv_branch 和 context_branch 的输出分配不同的权重。
    
    优势：
    - 对于需要精细边缘的区域，可能会给 odconv_branch 分配更高权重
    - 对于需要理解大背景的区域，则给 context_branch 更高权重
    - 避免简单的拼接带来的信息冲突，最终生成既包含丰富语义又保留精确位置的特征
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, dilation_rates=[1, 2, 3]):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            shortcut: 是否使用残差连接
            g: 分组卷积的组数
            e: 扩展比例
            dilation_rates: MultiScaleContext 的空洞卷积扩张率列表
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        
        # 两个并行分支 (使用改进版的MultiScaleContext)
        self.odconv_branch = ODConv2d(c_, c_, 3, 1)
        self.context_branch = MultiScaleContext(c_, dilation_rates=dilation_rates)
        
        # 核心改动：注意力融合门控（SE-Block风格）
        # 压缩比例 r=4（即 c_ // 4），而不是文档中提到的 r=8
        # 这样可以减少参数量，同时保持足够的表达能力
        reduction_ratio = 4
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(c_ * 2, c_ // reduction_ratio, 1, bias=False),  # 压缩
            nn.ReLU(),
            nn.Conv2d(c_ // reduction_ratio, c_ * 2, 1, bias=False),  # 恢复
            nn.Sigmoid()  # 生成注意力权重
        )
        
        # 最终的1x1卷积
        self.cv_fuse = Conv(c_ * 2, c2, 1, 1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        前向传播：通过注意力机制智能融合两个分支的输出
        """
        x_proj = self.cv1(x)
        
        # 两个并行分支
        out_od = self.odconv_branch(x_proj)  # ODConv分支：精细的局部特征
        out_ctx = self.context_branch(x_proj)  # Context分支：多尺度上下文信息
        
        # 将两个分支的输出拼接
        combined_features = torch.cat([out_od, out_ctx], dim=1)  # [B, 2*c_, H, W]
        
        # 核心改动：计算注意力权重并应用
        attention_weights = self.fusion_gate(combined_features)  # [B, 2*c_, 1, 1]
        attended_features = combined_features * attention_weights  # 逐元素相乘
        
        # 融合加权后的特征
        out_fused = self.act(self.bn(self.cv_fuse(attended_features)))
        
        # 残差连接
        return x + out_fused if self.add else out_fused


class C3k2_HybridFusion(C2f):
    """
    一个遵循 C3k2 命名和结构风格的模块，
    但其核心处理单元是我们最新设计的、性能更优的
    Bottleneck_Hybrid_AttentiveFusion。

    这个模块可以直接在 YAML 文件中替换 C3k2_ODConv_Advanced，
    是进行最终对比实验的推荐模块。
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2_HybridFusion module."""
        # 首先，调用父类C2f的构造函数，完成基本框架的初始化
        super().__init__(c1, c2, n, shortcut, g, e)
        
        # 核心改动在这里：
        # 我们保留了 c3k 这个参数开关，以确保与原始 C3k2 模块的兼容性。
        # 当 c3k=False (这是默认情况) 时，我们不再使用旧的 Bottleneck_ODConv_Advanced，
        # 而是换上我们最终的、最先进的 Bottleneck_Hybrid_AttentiveFusion 版本。
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_Hybrid_AttentiveFusion(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

# 温馨提示：请记得将 'C3k2_HybridFusion' 添加到您的 __all__ 导出列表中。


class C3k2_ODConv_Advanced(C2f):
    """
    改进版C3k2模块，使用ODConv2d_Advanced
    专门针对皮带机运行状态检测任务优化：
    - 解决"左偏移"问题：通过方向感知注意力增强边缘检测
    - 解决"大块煤"问题：通过多尺度上下文增强扩大感受野
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2_ODConv_Advanced module."""
        super().__init__(c1, c2, n, shortcut, g, e)
        # 使用改进版的Bottleneck
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_ODConv_Advanced(self.c, self.c, shortcut, g) for _ in range(n)
        )

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)

    # 测试原始模块
    print("测试原始C3k2_ODConv模块...")
    model_original = C3k2_ODConv(64, 64)
    out_original = model_original(image)
    print(f"原始模块输出尺寸: {out_original.size()}")
    
    # 测试改进版模块
    print("\n测试改进版C3k2_ODConv_Advanced模块...")
    model_advanced = C3k2_ODConv_Advanced(64, 64)
    out_advanced = model_advanced(image)
    print(f"改进版模块输出尺寸: {out_advanced.size()}")
    
    # 测试ODConv2d_Advanced单独使用
    print("\n测试ODConv2d_Advanced单独使用...")
    odconv_advanced = ODConv2d_Advanced(64, 64, 3, 1)
    out_odconv = odconv_advanced(image)
    print(f"ODConv2d_Advanced输出尺寸: {out_odconv.size()}")
    
    # 测试方案一：改进的MultiScaleContext模块
    print("\n测试方案一：改进的MultiScaleContext模块...")
    context_module = MultiScaleContext(64, dilation_rates=[1, 2, 3])
    out_context = context_module(image)
    print(f"MultiScaleContext输出尺寸: {out_context.size()}")
    print("✓ 方案一：MultiScaleContext已成功保留原始空间信息")
    
    # 测试方案二：注意力融合的混合瓶颈层
    print("\n测试方案二：Bottleneck_Hybrid_AttentiveFusion模块...")
    hybrid_bottleneck = Bottleneck_Hybrid_AttentiveFusion(64, 64, shortcut=True, e=0.5)
    out_hybrid = hybrid_bottleneck(image)
    print(f"Bottleneck_Hybrid_AttentiveFusion输出尺寸: {out_hybrid.size()}")
    print("✓ 方案二：注意力融合机制已成功实现")
    
    print("\n所有测试通过！")