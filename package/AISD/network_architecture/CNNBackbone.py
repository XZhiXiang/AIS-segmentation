# ------------------------------------------------------------------------
# CNN encoder
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight                      #
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)     #批量归一化，提升模型收敛速度和模型的精度
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)     ##续看
        weight = weight / std.expand_as(weight)  ###  /后面，吧std扩展为weight同样的维度大小
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)  ##对输入进行3D卷积
                     ##  x->输入张量的形状  weight->过滤器张量的形状  bias->可选偏置张良的形状  stride->卷积核步长  默认：1-padding-输入上隐含零填充  默认：0-groups-将输入分成组


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
    "3x3x3 convolution with padding"  #带填充的3x3x3卷积
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)


def Norm_layer(norm_cfg, inplanes):                        ##归一化层     防止梯度爆炸和梯度消失  卷积操作过后进去归一化层，再进入池化层
    if norm_cfg == 'BN':                                    ##选择归一化函数
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes,affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):      ##激活层  选择激活函数

    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv3d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ResBlock_CBAM(nn.Module):
    def __init__(self,in_places,  stride=1,downsampling=False, expansion = 4):
        super(ResBlock_CBAM,self).__init__()
        # self.expansion = expansion
        self.downsampling = downsampling

        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
        #     nn.BatchNorm2d(places),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
        #     nn.BatchNorm2d(places),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(places*self.expansion),
        # )
        self.cbam = CBAM(channel=in_places)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_places)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        # out = self.bottleneck(x)
        out = self.cbam(x)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResBlock(nn.Module):                    ##残差网络模块
    expansion = 1

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1, 1), downsample=None, weight_std=False):         ##构造函数
        super(ResBlock, self).__init__()

        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_std=weight_std)     ##卷积
        self.norm1 = Norm_layer(norm_cfg, planes)                                                                                ##归一化
        self.nonlin = Activation_layer(activation_cfg, inplace=True)                                                             ##激活函数
        self.downsample = downsample                                                                                             ##池化（下采样）

    def forward(self, x):

        residual = x

        out = self.conv1(x)                        #传入卷积层
        out = self.norm1(out)                      #传入归一化层

        if self.downsample is not None:            #传入池化层
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)                      #通过激活函数往后传递

        return out


class Backbone(nn.Module):

    arch_settings = {
        9: (ResBlock, (3, 3, 2))             ##字典，9是键，（ResBlock, (3, 3, 2)）为值
    }


    def __init__(self,                             ##构造函数
                 depth,
                 in_channels=1,                     #单通道
                 norm_cfg='BN',                      #归一化函数
                 activation_cfg='ReLU',              #激活函数
                 weight_std=False):
        super(Backbone, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth                                      #初始化
        block, layers = self.arch_settings[depth]
        self.inplanes = 64
        self.conv1 = conv3x3x3(in_channels, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False, weight_std=weight_std)         #卷积层
        self.norm1 = Norm_layer(norm_cfg, 64)                                  #归一化层
        self.nonlin = Activation_layer(activation_cfg, inplace=True)            #激活函数

        self.attention1 = ResBlock_CBAM(in_places=64)
        self.attention2 = ResBlock_CBAM(in_places=192)
        self.attention3 = ResBlock_CBAM(in_places=384)

        self.layer1 = self._make_layer(block, 192, layers[0], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, 384, layers[1], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, 384, layers[2], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layers = []

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):                    ##判断m的类型是否为Conv3d或者conv3d_wd
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')     #初始化权重
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=(1, 1, 1), norm_cfg='BN', activation_cfg='ReLU', weight_std=False):   #对每一层的层自定义，block是网络块，planes是通道数
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(   #一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
                conv3x3x3(
                    self.inplanes,                               #in_planes
                    planes * block.expansion,                    #out_planes
                    kernel_size=1,
                    stride=stride,
                    bias=False, weight_std=weight_std), Norm_layer(norm_cfg, planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, stride=stride, downsample=downsample, weight_std=weight_std))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)                #用1来填充m.weight
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)                  #用0填充

    def forward(self, x):
        out = []
        x = self.conv1(x)        #传入卷积层
        x = self.norm1(x)        #流入归一化层
        x = self.nonlin(x)       #再流入激活函数
        out.append(x)            #写入out列表

        x = self.attention1(x)

        x = self.layer1(x)
        out.append(x)

        x = self.attention2(x)

        x = self.layer2(x)
        out.append(x)

        x = self.attention3(x)

        x = self.layer3(x)
        out.append(x)

        return out

