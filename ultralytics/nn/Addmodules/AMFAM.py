import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ('Zoom_cat', 'DynamicScalSeq', 'Add', 'attention_model','Dy_Sample','ScalSeq', 'CategoryAwareDynamicASF')

#改进3 改了C'CategoryAwareDynamicASF还使用了Dy_Sample上采样

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv_l_post_down = Conv(in_dim, 2*in_dim, 3, 1, 1)

    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        # l = self.conv_l_post_down(l)
        # m = self.conv_m(m)
        # s = self.conv_s_pre_up(s)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        # s = self.conv_s_post_up(s)
        lms = torch.cat([l, m, s], dim=1)
        return lms

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class ScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(ScalSeq, self).__init__()
        self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x


class Dy_Sample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class DynamicScalSeq(nn.Module):    
    def __init__(self, inc, channel):
        super(DynamicScalSeq, self).__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel,1)
        self.conv1 =  Conv(inc[1], channel,1)
        self.conv2 =  Conv(inc[2], channel,1)
        self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))
        
        self.dysample1 = Dy_Sample(channel, 2, 'lp')
        self.dysample2 = Dy_Sample(channel, 4, 'lp')

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        if hasattr(self, 'conv0'):
            p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = self.dysample1(p4_2)
        p5_2 = self.conv2(p5)
        p5_2 = self.dysample2(p5_2)
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x


class Add(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()

    def forward(self, x):
        input1, input2 = x[0], x[1]
        x = input1 + input2
        return x


class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class attention_model(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)

    def forward(self, x):
        input1, input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x


class CategoryAwareDynamicASF(nn.Module):
    """
    动态类别感知ASF模块 (Category-Aware Dynamic ASF)
    实现三大核心改进机制：
    1. 类别引导的动态尺度权重生成
    2. 尺度间语义一致性约束
    3. 类别专属的双模式注意力增强
    4. 使用Dy_Sample进行自适应上采样
    """
    def __init__(self, inc, channel, num_classes=4):
        super(CategoryAwareDynamicASF, self).__init__()
        self.num_classes = num_classes
        self.channel = channel
        
        # 基础特征处理层
        self.conv_p3 = Conv(inc[0], channel, 1) if inc[0] != channel else nn.Identity()
        self.conv_p4 = Conv(inc[1], channel, 1)
        self.conv_p5 = Conv(inc[2], channel, 1)
        
        # 使用Dy_Sample进行自适应上采样
        self.dysample_p4 = Dy_Sample(channel, scale=2, style='lp', groups=4)
        self.dysample_p5 = Dy_Sample(channel, scale=4, style='lp', groups=4)
        
        # 机制1: 类别引导的动态尺度权重生成
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化获取每个尺度的全局特征
            Conv(channel + num_classes, 3, 1),  # 输出3个尺度的权重
            nn.Softmax(dim=1)  # 确保权重和为1
        )
        
        # 机制2: 尺度间语义一致性约束
        self.sim_conv = Conv(channel, channel, 1)  # 用于简化相似度计算
        
        # 机制3: 类别专属的双模式注意力增强
        self.channel_att = channel_att(channel)  # 适合大块煤的通道注意力
        self.spatial_att = local_att(channel)   # 适合边缘的空间注意力
        
        # 输出层优化
        self.output_conv = Conv(channel, channel, 1)
        
        # 初始化权重生成器
        self._init_weight_generator()
    
    def _init_weight_generator(self):
        """初始化权重生成器，避免初始权重偏差过大"""
        for m in self.weight_generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, class_emb=None, conf_maps=None):
        """
        Args:
            x: 多尺度特征 [P3, P4, P5]
            class_emb: 类别嵌入向量 [B, num_classes] (可选)
            conf_maps: 各尺度的类别置信度图 (可选)
        """
        p3, p4, p5 = x[0], x[1], x[2]
        
        # 基础特征处理
        p3_feat = self.conv_p3(p3)
        p4_feat = self.conv_p4(p4)
        p5_feat = self.conv_p5(p5)
        
        # 使用Dy_Sample进行自适应上采样到P3的尺度
        p4_feat = self.dysample_p4(p4_feat)  # 2倍上采样
        p5_feat = self.dysample_p5(p5_feat)  # 4倍上采样
        
        # 机制1: 类别引导的动态尺度权重生成
        if class_emb is not None:
            # 获取每个尺度的全局特征
            p3_global = F.adaptive_avg_pool2d(p3_feat, 1)  # [B, channel, 1, 1]
            p4_global = F.adaptive_avg_pool2d(p4_feat, 1)
            p5_global = F.adaptive_avg_pool2d(p5_feat, 1)
            
            # 将全局特征与类别嵌入拼接
            p3_with_class = torch.cat([p3_global, class_emb.unsqueeze(-1).unsqueeze(-1)], dim=1)
            p4_with_class = torch.cat([p4_global, class_emb.unsqueeze(-1).unsqueeze(-1)], dim=1)
            p5_with_class = torch.cat([p5_global, class_emb.unsqueeze(-1).unsqueeze(-1)], dim=1)
            
            # 生成动态权重
            weight_p3 = self.weight_generator(p3_with_class)  # [B, 3, 1, 1]
            weight_p4 = self.weight_generator(p4_with_class)
            weight_p5 = self.weight_generator(p5_with_class)
            
            # 动态加权
            p3_weighted = p3_feat * weight_p3[:, 0:1]  # 取第一个权重
            p4_weighted = p4_feat * weight_p4[:, 1:2]   # 取第二个权重
            p5_weighted = p5_feat * weight_p5[:, 2:3]   # 取第三个权重
        else:
            # 如果没有类别信息，使用均等权重
            p3_weighted = p3_feat
            p4_weighted = p4_feat
            p5_weighted = p5_feat
        
        # 机制2: 尺度间语义一致性约束
        if conf_maps is not None:
            # 计算尺度相似度掩码
            sim_p3_p4 = F.cosine_similarity(p3_weighted, p4_weighted, dim=1).unsqueeze(1)
            sim_p3_p5 = F.cosine_similarity(p3_weighted, p5_weighted, dim=1).unsqueeze(1)
            
            # 激活相似度掩码
            sim_mask_p3_p4 = torch.sigmoid(sim_p3_p4 * 5)
            sim_mask_p3_p5 = torch.sigmoid(sim_p3_p5 * 5)
            
            # 冲突抑制融合
            conflict_region_p4 = sim_mask_p3_p4 < 0.5
            conflict_region_p5 = sim_mask_p3_p5 < 0.5
            
            # 在冲突区域以高置信度尺度为主
            p4_fused = torch.where(conflict_region_p4, 
                                  p3_weighted * 0.1 + p4_weighted * 0.9, 
                                  p3_weighted + p4_weighted)
            p5_fused = torch.where(conflict_region_p5,
                                  p3_weighted * 0.1 + p5_weighted * 0.9,
                                  p3_weighted + p5_weighted)
            
            # 最终融合
            fused_feat = (p3_weighted + p4_fused + p5_fused) / 3
        else:
            # 简单融合
            fused_feat = p3_weighted + p4_weighted + p5_weighted
        
        # 机制3: 类别专属的双模式注意力增强
        if class_emb is not None:
            # 根据类别动态切换注意力模式
            dominant_class = class_emb.argmax(dim=1)
            
            # 假设类别2是大块煤，其他是边缘
            is_large_coal = dominant_class.eq(2)
            
            if is_large_coal.any():
                # 大块煤：使用通道注意力
                att_mask = self.channel_att(fused_feat)
            else:
                # 边缘：使用空间注意力
                att_mask = self.spatial_att(fused_feat)
            
            fused_feat = fused_feat * att_mask
        
        # 输出层优化：残差连接 + 通道压缩
        if isinstance(self.conv_p3, nn.Identity):
            # 如果P3没有经过卷积，直接使用原始P3进行残差连接
            fused_feat = fused_feat + p3
        else:
            fused_feat = fused_feat + p3_feat
        
        # 最终输出
        output = self.output_conv(fused_feat)
        
        return output