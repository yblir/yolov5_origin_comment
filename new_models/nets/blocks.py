# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings

import torch
import torch.nn as nn


# ============================================= 核心模块 =====================================
def autopad(k, p=None):
    """
    为same卷积或same池化作自动扩充（0填充）  Pad to 'same'
    :params k: 卷积核的kernel_size
    :return p: 自动计算的需要pad值（0填充）
    """
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)  # 自动计算pad数
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, ch_in, ch_out, k=1, s=1, p=None, g=1,
                 act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        """
        Standard convolution  conv+BN+act
        :params ch_in: 输入的channel值
        :params ch_out: 输出的channel值
        :params k: 卷积的kernel_size
        :params s: 卷积的stride
        :params p: 卷积的padding  一般是None  可以通过autopad自行计算需要pad的padding数
        :params g: 卷积的groups数  =1就是普通的卷积  >1就是深度可分离卷积
        :params act: 激活函数类型   True就是SiLU()/Swish   False就是不使用激活函数
                     类型是nn.Module就使用传进来的激活函数类型
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(ch_out,eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # 网络的定义次序优__init__决定,也是打印模型的次序. 传播次序由forward决定
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        前向融合计算  减少推理时间
        """
        return self.act(self.conv(x))


class DWConv(Conv):
    # 没有使用
    # Depth-wise convolution class
    def __init__(self, ch_in, ch_out, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(ch_in, ch_out, k, s, g=math.gcd(ch_in, ch_out), act=act)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fch_in = nn.Linear(c, c, bias=False)
        self.fc12 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc12(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, ch_in, ch_out, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if ch_in != ch_out:
            self.conv = Conv(ch_in, ch_out)
        self.linear = nn.Linear(ch_out, ch_out)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(ch_out, num_heads) for _ in range(num_layers)))
        self.ch_out = ch_out

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.ch_out, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, ch_in, ch_out, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        """
        Standard bottleneck  Conv+Conv+shortcut
        :params ch_in: 第一个卷积的输入channel
        :params ch_out: 第二个卷积的输出channel
        :params shortcut: bool 是否有shortcut连接 默认是True
        :params g: 卷积分组的个数  =1就是普通卷积  >1就是深度可分离卷积
        :params e: expansion ratio  e*ch_out就是第一个卷积的输出channel=第二个卷积的输入channel
        """
        super(Bottleneck, self).__init__()

        c_ = int(ch_out * e)  # hidden channels
        self.cv1 = Conv(ch_in, c_, 1, 1)  # 1x1
        self.cv2 = Conv(c_, ch_out, 3, 1, g=g)  # 3x3
        self.add = shortcut and ch_in == ch_out  # shortcut=True and ch_in == ch_out 才能做shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, ch_in, ch_out, n=1, shortcut=True, g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        """
        CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        :params ch_in: 整个BottleneckCSP的输入channel
        :params ch_out: 整个BottleneckCSP的输出channel
        :params n: 有n个Bottleneck
        :params shortcut: bool Bottleneck中是否有shortcut，默认True
        :params g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
        :params e: expansion ratio ch_outxe=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(ch_out * e)  # hidden channels
        self.cv1 = Conv(ch_in, c_, 1, 1)
        self.cv2 = nn.Conv2d(ch_in, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, ch_out, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        # 叠加n次Bottleneck
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions 跨阶段局部网络
    # 输入没做split, 而是来自同样的输入,这与yolov4的cpsnet不同
    def __init__(self, ch_in, ch_out, n=1, shortcut=True, g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        """
        CSP Bottleneck with 3 convolutions
        :params ch_in: 整个BottleneckCSP的输入channel
        :params ch_out: 整个BottleneckCSP的输出channel
        :params n: 有n个Bottleneck
        :params shortcut: bool Bottleneck中是否有shortcut，默认True
        :params g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
        :params e: expansion ratio ch_outxe=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
        """
        super(C3, self).__init__()
        c_ = int(ch_out * e)  # hidden channels
        # cv1经过多次bottleneck
        self.cv1 = Conv(ch_in, c_, 1, 1)
        # cv2直连
        self.cv2 = Conv(ch_in, c_, 1, 1)
        self.cv3 = Conv(2 * c_, ch_out, 1)  # act=FReLU(ch_out)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # 实验性 CrossConv
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)),
                                   self.cv2(x)
                                   ), dim=1))


class C3TR(C3):
    """
    这部分是根据上面的C3结构改编而来的, 将原先的Bottleneck替换为调用TransformerBlock模块
    """

    # C3 module with TransformerBlock()
    def __init__(self, ch_in, ch_out, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(ch_in, ch_out, n, shortcut, g, e)
        c_ = int(ch_out * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, ch_in, ch_out, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(ch_in, ch_out, n, shortcut, g, e)
        c_ = int(ch_out * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, ch_in, ch_out, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(ch_in, ch_out, n, shortcut, g, e)
        c_ = int(ch_out * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # 空间金字塔池化,增大感受野,提取最重要的上下文特征
    # 做填充,使得池化前后特征图尺寸不变
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, ch_in, ch_out, k=(5, 9, 13)):
        super().__init__()
        c_ = ch_in // 2  # hidden channels
        self.cv1 = Conv(ch_in, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), ch_out, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, ch_in, ch_out, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = ch_in // 2  # hidden channels
        self.cv1 = Conv(ch_in, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, ch_out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # 减伤flops,提高速度,对精度没有影响
    # Focus wh information into c-space
    def __init__(self, ch_in, ch_out, k=1, s=1, p=None, g=1,
                 act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        """
        理论：从高分辨率图像中，周期性的抽出像素点重构到低分辨率图像中，即将图像相邻的四个位置进行堆叠，
            聚焦wh维度信息到c通道空，提高每个点感受野，并减少原始信息的丢失，该模块的设计主要是减少计算量加快速度。
        Focus wh information into c-space 把宽度w和高度h的信息整合到c空间中
        先做4个slice 再concat 最后再做Conv
        slice后 (b,ch_in,w,h) -> 分成4个slice 每个slice(b,ch_in,w/2,h/2)
        concat(dim=1)后 4个slice(b,ch_in,w/2,h/2)) -> (b,4ch_in,w/2,h/2)
        conv后 (b,4ch_in,w/2,h/2) -> (b,ch_out,w/2,h/2)
        :params ch_in: slice后的channel
        :params ch_out: Focus最终输出的channel
        :params k: 最后卷积的kernel
        :params s: 最后卷积的stride
        :params p: 最后卷积的padding
        :params g: 最后卷积的分组情况  =1普通卷积  >1深度可分离卷积
        :params act: bool激活函数类型  默认True:SiLU()/Swish  False:不用激活函数
        """
        super(Focus, self).__init__()
        self.conv = Conv(ch_in * 4, ch_out, k, s, p, g, act)  # concat后的卷积（最后的卷积）
        # self.contract = Contract(gain=2)  # 也可以调用Contract函数实现slice操作

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # x(b,c,w,h) -> y(b,4c,w/2,h/2)  有点像做了个下采样
        return self.conv(torch.cat((x[..., ::2, ::2],
                                    x[..., 1::2, ::2],
                                    x[..., ::2, 1::2],
                                    x[..., 1::2, 1::2]), dim=1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, ch_in, ch_out, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = ch_out // 2  # hidden channels
        self.cv1 = Conv(ch_in, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), dim=1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, ch_in, ch_out, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = ch_out // 2
        self.conv = nn.Sequential(GhostConv(ch_in, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, ch_out, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(ch_in, ch_in, k, s, act=False),
                                      Conv(ch_in, ch_out, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
