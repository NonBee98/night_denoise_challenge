import math

import torch
import torch.nn as nn

from .common_blocks import *

__all__ = ['ghostnet']


class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(in_channels,
                   out_channels,
                   kernel_size=3,
                   stride=1,
                   LeakyReLU=False):
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  kernel_size // 2,
                  groups=in_channels,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True) if LeakyReLU else nn.Sequential(),
    )


class GhostModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 ratio=2,
                 dw_size=3,
                 stride=1,
                 LeakyReLU=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      init_channels,
                      kernel_size,
                      stride,
                      kernel_size // 2,
                      bias=False),
            nn.BatchNorm2d(init_channels),
            nn.LeakyReLU(inplace=True) if LeakyReLU else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels,
                      new_channels,
                      dw_size,
                      1,
                      dw_size // 2,
                      groups=init_channels,
                      bias=False),
            nn.BatchNorm2d(new_channels),
            nn.LeakyReLU(inplace=True) if LeakyReLU else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):

    def __init__(self, in_channels, hidden_dim, out_channels, kernel_size,
                 stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(in_channels, hidden_dim, kernel_size=1,
                        LeakyReLU=True),
            # dw
            depthwise_conv(
                hidden_dim, hidden_dim, kernel_size, stride, LeakyReLU=False)
            if stride == 2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim,
                        out_channels,
                        kernel_size=1,
                        LeakyReLU=False),
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(in_channels,
                               in_channels,
                               kernel_size,
                               stride,
                               LeakyReLU=False),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class GhostNet(nn.Module):

    def __init__(self, cfgs, in_channels=3, scale=1.):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        output_channel = make_divisible(32 * scale, 4)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, output_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channel), nn.LeakyReLU(inplace=True))
        input_channel = output_channel
        # building inverted residual blocks

        block = GhostBottleneck

        stage_1 = []
        for k, exp_size, c, use_se, s in self.cfgs[0]:
            output_channel = make_divisible(c * scale, 4)
            hidden_channel = make_divisible(exp_size * scale, 4)
            stage_1.append(
                block(input_channel, hidden_channel, output_channel, k, s,
                      use_se))
            input_channel = output_channel

        stage_2 = []
        for k, exp_size, c, use_se, s in self.cfgs[1]:
            output_channel = make_divisible(c * scale, 4)
            hidden_channel = make_divisible(exp_size * scale, 4)
            stage_2.append(
                block(input_channel, hidden_channel, output_channel, k, s,
                      use_se))
            input_channel = output_channel

        stage_3 = []
        for k, exp_size, c, use_se, s in self.cfgs[2]:
            output_channel = make_divisible(c * scale, 4)
            hidden_channel = make_divisible(exp_size * scale, 4)
            stage_3.append(
                block(input_channel, hidden_channel, output_channel, k, s,
                      use_se))
            input_channel = output_channel

        self.stage1 = nn.Sequential(*stage_1)
        self.stage2 = nn.Sequential(*stage_2)
        self.stage3 = nn.Sequential(*stage_3)

        # self._initialize_weights()

    def forward(self, x):
        stem_out = self.stem(x)
        stage1_out = self.stage1(stem_out)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)

        return stem_out, stage1_out, stage2_out, stage3_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    # k, hidden, out, SE, stride
    cfgs = [
        [
            [3, 96, 64, 0, 2],  #  32 * 32
            [3, 256, 128, 1, 1]
        ],  #  32 * 32
        [
            [3, 256, 128, 0, 2],  #  16 * 16
            [3, 256, 128, 0, 1],
            [3, 256, 128, 0, 1],
            [3, 256, 128, 0, 1],
            [3, 256, 128, 0, 1],
            [3, 512, 256, 1, 1]
        ],
        [
            [5, 256, 128, 0, 2],  # 8 * 8
            [5, 512, 128, 0, 1],
            [5, 512, 128, 0, 1],
            [5, 512, 128, 0, 1],
            [5, 960, 512, 1, 1]
        ]
    ]

    return GhostNet(cfgs, **kwargs)
