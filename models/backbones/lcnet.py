import torch
import torch.nn as nn
import torch.nn.functional as F
from .common_blocks import *

NET_CONFIG = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}


class ConvBNLayer(nn.Module):

    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=num_channels,
                              out_channels=num_filters,
                              kernel_size=filter_size,
                              stride=stride,
                              padding=(filter_size - 1) // 2,
                              groups=num_groups,
                              bias=False)

        self.bn = nn.BatchNorm2d(num_filters)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class DepthwiseSeparable(nn.Module):

    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(num_channels=num_channels,
                                   num_filters=num_channels,
                                   filter_size=dw_size,
                                   stride=stride,
                                   num_groups=num_channels)
        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(num_channels=num_channels,
                                   filter_size=1,
                                   num_filters=num_filters,
                                   stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class PPLCNet(nn.Module):

    def __init__(self,
                 scale=1.0,
                 num_classes=3,
                 dropout_prob=0.2,
                 class_expand=1280,
                 return_patterns=None,
                 return_stages=None):
        super(PPLCNet, self).__init__()
        self.scale = scale
        self.class_expand = class_expand

        self.conv1 = ConvBNLayer(num_channels=3,
                                 filter_size=3,
                                 num_filters=make_divisible(16 * scale),
                                 stride=2)

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(num_channels=make_divisible(in_c * scale),
                               num_filters=make_divisible(out_c * scale),
                               dw_size=k,
                               stride=s,
                               use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(num_channels=make_divisible(in_c * scale),
                               num_filters=make_divisible(out_c * scale),
                               dw_size=k,
                               stride=s,
                               use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(num_channels=make_divisible(in_c * scale),
                               num_filters=make_divisible(out_c * scale),
                               dw_size=k,
                               stride=s,
                               use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(num_channels=make_divisible(in_c * scale),
                               num_filters=make_divisible(out_c * scale),
                               dw_size=k,
                               stride=s,
                               use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(num_channels=make_divisible(in_c * scale),
                               num_filters=make_divisible(out_c * scale),
                               dw_size=k,
                               stride=s,
                               use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks6"])
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.last_conv = nn.Conv2d(in_channels=make_divisible(
            NET_CONFIG["blocks6"][-1][2] * scale),
                                   out_channels=self.class_expand,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=False)

        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(self.class_expand, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)

        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)

        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()