import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_blocks import *


# Basic conv layer
class Conv(nn.Module):

    def __init__(
            self,
            c1,  # in channels
            c2,  # out channels 
            k=1,  # kernel size 
            s=1,  # padding
            p=0,  # padding
            d=1,  # dilation
            act_type='silu',  # activation
            depthwise=False):
        super(Conv, self).__init__()
        convs = []
        if depthwise:
            # depthwise conv
            convs.append(
                nn.Conv2d(c1,
                          c1,
                          kernel_size=k,
                          stride=s,
                          padding=p,
                          dilation=d,
                          groups=c1,
                          bias=False))
            convs.append(nn.BatchNorm2d(c1))
            if act_type is not None:
                if act_type == 'silu':
                    convs.append(nn.SiLU(inplace=True))
                elif act_type == 'lrelu':
                    convs.append(nn.LeakyReLU(0.1, inplace=True))

            # pointwise conv
            convs.append(
                nn.Conv2d(c1,
                          c2,
                          kernel_size=1,
                          stride=s,
                          padding=0,
                          dilation=d,
                          groups=1,
                          bias=False))
            convs.append(nn.BatchNorm2d(c2))
            if act_type is not None:
                if act_type == 'silu':
                    convs.append(nn.SiLU(inplace=True))
                elif act_type == 'lrelu':
                    convs.append(nn.LeakyReLU(0.1, inplace=True))

        else:
            convs.append(
                nn.Conv2d(c1,
                          c2,
                          kernel_size=k,
                          stride=s,
                          padding=p,
                          dilation=d,
                          groups=1,
                          bias=False))
            convs.append(nn.BatchNorm2d(c2))
            if act_type is not None:
                if act_type == 'silu':
                    convs.append(nn.SiLU(inplace=True))
                elif act_type == 'lrelu':
                    convs.append(nn.LeakyReLU(0.1, inplace=True))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


class ELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's backbone
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 model_size='large',
                 act_type='silu',
                 depthwise=False):
        super(ELANBlock, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        if model_size == 'large':
            depth = 2
        elif model_size == 'tiny':
            depth = 1
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv3 = nn.Sequential(*[
            Conv(inter_dim,
                 inter_dim,
                 k=3,
                 p=1,
                 act_type=act_type,
                 depthwise=depthwise) for _ in range(depth)
        ])
        self.cv4 = nn.Sequential(*[
            Conv(inter_dim,
                 inter_dim,
                 k=3,
                 p=1,
                 act_type=act_type,
                 depthwise=depthwise) for _ in range(depth)
        ])

        self.out = Conv(inter_dim * 4, out_dim, k=1)

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, out_dim, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # [B, C, H, W] -> [B, out_dim, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

        return out


class DownSample(nn.Module):

    def __init__(self, in_dim, act_type='silu'):
        super().__init__()
        inter_dim = in_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type))

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out


# ELANNet of YOLOv7
class ELANNet(nn.Module):
    """
    ELAN-Net of YOLOv7.
    """

    def __init__(self,
                 in_channels=3,
                 depthwise=False,
                 stage_channels=[128, 256, 512],
                 act_type='lrelu'):
        super(ELANNet, self).__init__()

        self.stage_channels = stage_channels

        self.stem = Conv(in_channels,
                         32,
                         k=3,
                         p=1,
                         act_type=act_type,
                         depthwise=depthwise)
        self.layer_1 = nn.Sequential(
            Conv(32, 64, k=3, p=1, s=2, act_type=act_type,
                 depthwise=depthwise),
            Conv(64,
                 stage_channels[0],
                 k=3,
                 p=1,
                 act_type=act_type,
                 depthwise=depthwise)  # P1/2
        )
        self.layer_2 = nn.Sequential(
            Conv(stage_channels[0],
                 stage_channels[0] * 2,
                 k=3,
                 p=1,
                 s=2,
                 act_type=act_type,
                 depthwise=depthwise),
            ELANBlock(stage_channels[0] * 2,
                      out_dim=stage_channels[1],
                      expand_ratio=0.5,
                      act_type=act_type,
                      depthwise=depthwise)  # P2/4
        )
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=stage_channels[1], act_type=act_type),
            ELANBlock(in_dim=stage_channels[1],
                      out_dim=stage_channels[2],
                      expand_ratio=0.5,
                      act_type=act_type,
                      depthwise=depthwise)  # P3/8
        )

    def forward(self, x):
        stem_out = self.stem(x)
        stage1_out = self.layer_1(stem_out)
        stage2_out = self.layer_2(stage1_out)
        stage3_out = self.layer_3(stage2_out)

        return stem_out, stage1_out, stage2_out, stage3_out
