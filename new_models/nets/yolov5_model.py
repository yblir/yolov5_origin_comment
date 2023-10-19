import torch
import torch.nn as nn
import numpy as np
from blocks import Focus, Conv, C3, Bottleneck, BottleneckCSP, SPP, SPPF


class CSPDarknet(nn.Module):
    def __init__(self, base_ch, base_depth):
        super(CSPDarknet, self).__init__()
        self.stem = Focus(3, base_ch, k=3)
        self.dark2 = nn.Sequential(Conv(base_ch, base_ch * 2, k=3, s=2),
                                   C3(base_ch * 2, base_ch * 2, n=base_depth))
        self.dark3 = nn.Sequential(Conv(base_ch * 2, base_ch * 4, k=3, s=2),
                                   C3(base_ch * 4, base_ch * 4, n=base_depth * 2))
        self.dark4 = nn.Sequential(Conv(base_ch * 4, base_ch * 8, k=3, s=2),
                                   C3(base_ch * 8, base_ch * 8, n=base_depth * 3))

        self.dark5 = nn.Sequential(Conv(base_ch * 8, base_ch * 16, k=3, s=2),
                                   # SPP(base_ch * 16, base_ch * 16),
                                   C3(base_ch * 16, base_ch * 16, n=base_depth, shortcut=False))

    def forward(self, inputs):
        x = self.stem(inputs)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = x  # 256,80,80
        x = self.dark4(x)
        feat2 = x  # 512,40,40
        x = self.dark5(x)
        feat3 = x  # 1024,20,20
        return feat1, feat2, feat3


class Neck(nn.Module):
    '''PAnet,路径聚合网络'''

    def __init__(self, base_ch, base_depth):
        super(Neck, self).__init__()
        # self.sppf = SPPF(base_ch * 16, base_ch * 8, k=1)
        # self.conv1 = Conv(base_ch * 16, base_ch * 8, k=1, s=1)
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 
        # self.c3_1 = C3(base_ch * 16, base_ch * 4, n=base_depth)
        # 
        # self.conv2 = Conv(base_ch * 8, base_ch * 2, k=1, s=1)
        # self.c3_2 = C3(base_ch * 8, base_ch * 4, n=base_depth)
        # 
        # self.conv3 = Conv(base_ch * 4, base_ch * 4, k=3, s=2)
        # self.c3_3 = C3(base_ch * 8, base_ch * 4, n=base_depth)
        # 
        # self.conv4 = Conv(base_ch * 8, base_ch * 4, k=3, s=2)
        # self.c3_4 = C3(base_ch * 16, base_ch * 8, n=base_depth)
        self.stage9 = SPPF(base_ch * 16, base_ch * 16)
        self.stage10 = Conv(base_ch * 16, base_ch * 8, k=1, s=1)
        self.stage11 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.stage12=nn.Concat()
        self.stage13 = C3(base_ch * 16, base_ch * 8, n=base_depth)

        # 40, 40, 512 -> 40, 40, 256
        self.stage14 = Conv(base_ch * 8, base_ch * 4, k=1, s=1)
        self.stage15 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.stage16=nn.Concat()
        self.stage17 = C3(base_ch * 8, base_ch * 4, n=base_depth)

        # 下采样阶段,使用卷积完成下采样过程. 有可能会用可分离卷积?
        self.stage18 = Conv(base_ch * 4, base_ch * 4, k=3, s=2)
        # self.stage19=nn.Concat()
        # 下采样阶段的csp
        self.stage20 = C3(base_ch * 8, base_ch * 8, n=base_depth)
        self.stage21 = Conv(base_ch * 8, base_ch * 8, k=3, s=2)
        # self.stage22=nn.Concat()
        # 40, 40, 1024 -> 40, 40, 512
        self.stage23 = C3(base_ch * 16, base_ch * 16, n=base_depth)

    def forward(self, inputs):
        feat1, feat2, feat3 = inputs
        P5 = self.last_conv(feat3)  # 1024,20,20 => 512,20,20
        P5_upsample = self.upsample(P5)  # 512,20,20 => 512,40,40

        P5_P4_cat = torch.cat([P5_upsample, feat2], dim=1)  # P5的上采样与dark4的输出拼接, =>1024,40,40
        P4 = self.P4_P3_csp(P5_P4_cat)  # 拼接后还有经过csp模块,1024,40,40 => 512,40,40
        P4 = self.P4_conv(P4)  # 512,40,40 => 256,40,40

        P4_upsample = self.upsample(P4)  # 256,80,80
        P4_P3_cat = torch.cat([P4_upsample, feat1], dim=1)  # =>512,80,80

        P3_out = self.P3_csp(P4_P3_cat)  # 加强网络最终输出之一 512,80,80 => 256,80,80

        P3_down = self.down_conv1(P3_out)  # 下采样, 256,80,80 => 256,40,40
        P3_P4_cat = torch.cat([P3_down, P4], dim=1)  # 下采样阶段的拼接, =>512,40,40
        P4_out = self.P3_P4_csp(P3_P4_cat)  # 加强网络最终输出之一 512,40,40 => 512,40,40

        P4_down = self.down_conv2(P4_out)  # 512,40,40 => 512,20,20
        P4_P5_cat = torch.cat([P4_down, P5], dim=1)  # => 1024,20,20

        P5_out = self.P4_P5_csp(P4_P5_cat)  # 加强网络最终输出之一 1024,20,20 => 1024,20,20

        # shape分别为(256,80,80),(512,40,40),(1024,20,20)
        return P3_out, P4_out, P5_out


class Head(nn.Module):
    def __init__(self, ch_in=[128, 256, 512], num_class=80):
        super(Head, self).__init__()
        ch_out = 3 * (num_class + 5)
        self.m = nn.ModuleList([nn.Conv2d(i, ch_out, kernel_size=(1, 1), stride=(1, 1)) for i in ch_in])

    def forward(self, inputs):
        return self.m(inputs)


class YoloBody(nn.Module):
    def __init__(self, phi):
        super(YoloBody, self).__init__()
        depth_dict = {'s': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
        width_dict = {'s': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        self.darknet = CSPDarknet(32, 1)
        self.neck = Neck(32, 1)
        self.head = Head()

    def forward(self, inputs):
        x = self.darknet(inputs)  # dark输出是一个列表，包含三个不同尺度的feature
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    model = YoloBody('l')
    pre_dict = torch.load(r'F:\GtiEE\my_yolov5_pytorch\yolov5s.pt')
    pre_dict=pre_dict['model']
    # print(type(pre_dict))
    for k,v in pre_dict.items():
        print(k)
    dic= model.state_dict()
    for k,v in dic.items():
        print(k)
    # model=model.load_state_dict(pre_dic)
    # print(model)

    model_dict = model.state_dict()
    # 此时model_list保存的都是键名,没有值
    model_list = list(model_dict)
    # 加载预训练权重,也是一个有序字典
    # pre_dict = torch.load(yaml_cfg['model_path'], map_location=device)
    # 重新更新预训练模型,因为自己搭建的模型部分属性名与原始模型不同,所以不能直接加载,需要把预训练的键名替换成自己的
    # 以下是根据每层参数的shape替换原来的键名.如果构建的模型层次序或shape与原始模型不一致, 莫得法,神仙也搞不定~
    pre_dict = {model_list[i]: v for i, (k, v) in enumerate(pre_dict.items())
                if np.shape(model_dict[model_list[i]]) == np.shape(v)}
    # 使用更新后的预训练权重,更新本模型权重
    model_dict.update(pre_dict)
    # 加载模型权重字典
    model.load_state_dict(model_dict)
    print(model)
