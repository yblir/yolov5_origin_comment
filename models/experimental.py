# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
import math

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv
from utils.downloads import attempt_download
# todo yolov5最新版没有这个模块
class GhostConv(nn.Module):
    """
    Ghost Convolution 幻象卷积  轻量化网络卷积模块
    论文: https://arxiv.org/abs/1911.11907
    源码: https://github.com/huawei-noah/ghostnet
    常见的几种使用方式: https://github.com/ultralytics/yolov5/issues/2905
    """
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        # 第一步卷积: 少量卷积, 一般是一半的计算量
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        # 第二步卷积: cheap operations 使用3x3或5x5的卷积, 并且是逐个特征图的进行卷积（Depth-wise convolutional）
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)
class GhostBottleneck(nn.Module):
    """
    Ghost Convolution 幻象卷积  轻量化网络卷积模块
    论文: https://arxiv.org/abs/1911.11907
    源码: https://github.com/huawei-noah/ghostnet
    """
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        # 注意, 源码中并不是直接Identity连接, 而是先经过一个DWConv + Conv, 再进行shortcut连接的。
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

# todo yolov5最新版没有这个模块
class CrossConv(nn.Module):
    """可以用在C3模块中(实验)
    Cross Convolution Downsample   3x3 -> 1x9 + 9x1
    https://github.com/ultralytics/yolov5/issues/4030
    """
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        # 1x5+5x1  或1x3+3x1  可以多多尝试
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    """
    加权特征融合: 学习不同输入特征的重要性，对不同输入特征有区分的融合  Weighted sum of 2 or more layers
    思想: 传统的特征融合往往只是简单的feature map叠加/相加 (sum them up), 比如使用concat或者shortcut连接, 而不对同时加进来的
         feature map进行区分。然而,不同的输入feature map具有不同的分辨率, 它们对融合输入feature map的贡献也是不同的, 因此简单
         的对他们进行相加或叠加处理并不是最佳的操作, 所以这里我们提出了一种简单而高效的加权特融合的机制。
    from: https://arxiv.org/abs/1911.09070
    """
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # 是否使用加权权重融合
        self.iter = range(n - 1)  # 加权 iter
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # # 初始化可学习权重

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2   # 得到每一个layer的可学习权重
            for i in self.iter:
                y = y + x[i + 1] * w[i]     # 加权特征融合
        else:
            for i in self.iter:
                y = y + x[i + 1]            # 特征融合
        return y


class MixConv2d(nn.Module):
    """
    Mixed Depthwise Conv 混合深度卷积 就是使用不同大小的卷积核对深度卷积的不同channel分组处理 也可以看作是分组深度卷积 + Inception结构的多种卷积核混用
    论文: https://arxiv.org/abs/1907.09595.
    源码: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet.
    """

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        """
        :params c1: 输入feature map的通道数
        :params c2: 输出的feature map的通道数（这个函数的关键点就是对c2进行分组）
        :params k: 混合的卷积核大小 其实论文里是[3, 5, 7...]用的比较多的
        :params s: 步长 stride
        :params equal_ch: 通道划分方式 有均等划分和指数划分两种方式  默认是均等划分
        """
        super(MixConv2d, self).__init__()
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # 均等划分通道
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # 指数划分通道
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """
    模型集成  Ensemble of models
    动机: 减少模型的泛化误差
    https://github.com/ultralytics/yolov5/issues/318
    来源: https://www.sciencedirect.com/topics/computer-science/ensemble-modeling
    """
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble	求两个模型结果的最大值
        # y = torch.stack(y).mean(0)  # mean ensemble	求两个模型结果的均值
        y = torch.cat(y, 1)  # nms ensemble				将两个模型结果concat 后面做nms(等于翻了一倍的pred)
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """用在val.py、detect.py、train.py等文件中  一般用在测试、验证阶段
    加载模型权重文件并构建模型（可以构造普通模型或者集成模型）
    Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    :params weights: 模型的权重文件地址 默认weights/yolov5s.pt
                     可以是[a]也可以是list格式[a, b]  如果是list格式将调用上面的模型集成函数 多模型运算 提高最终模型的泛化误差
    :params device: attempt_download函数参数  表示模型运行设备device
    :params inplace: pytorch 1.7.0 compatibility设置
    """
    from models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

    # Compatibility updates(关于版本兼容的设置)
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    if len(model) == 1:  # 单个模型 正常返回
        return model[-1]  # return model
    # 多个模型 使用模型集成 并对模型先进行一些必要的设置
    print(f'Ensemble created with {weights}\n')
        # 给每个模型一个name属性
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
        # 给每个模型分配stride属性
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model  # 返回集成模型
