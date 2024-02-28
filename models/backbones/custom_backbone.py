import torch
import torch.nn as nn

from .common_blocks import *

__all__ = ['CustomELANNet', 'CustomCSPDNet']


class CustomELANNet(nn.Module):
    """
    Modified ELAN-Net of YOLOv7.
    """

    def __init__(self,
                 in_channels=3,
                 residual=True,
                 repconv=True,
                 stage_channels=[64, 128, 256, 256],
                 act_type='lrelu',
                 depthwise=False,
                 **args):
        super(CustomELANNet, self).__init__()
        self.in_channels = in_channels
        self.residual = residual
        self.repconv = repconv
        self.stage_channels = stage_channels
        self.act_type = act_type

        basic_block = CustomResidualELANBlock if residual else CustomELANBlock

        self.stem = nn.Sequential(
            RepConv(in_channels, 32, identity=False, **args)
            if repconv else Conv(in_channels,
                                 32,
                                 k=3,
                                 p=1,
                                 act_type=act_type,
                                 depthwise=depthwise),
            basic_block(32,
                        out_dim=stage_channels[0],
                        expand_ratio=0.5,
                        act_type=act_type,
                        depthwise=depthwise,
                        repconv=repconv,
                        model_size='large',
                        **args))
        self.layer_1 = nn.Sequential(
            DownSample(stage_channels[0], act_type=act_type),
            basic_block(stage_channels[0],
                        out_dim=stage_channels[1],
                        expand_ratio=0.5,
                        act_type=act_type,
                        depthwise=depthwise,
                        repconv=repconv,
                        model_size='large',
                        **args)  # P1/2
        )
        self.layer_2 = nn.Sequential(
            DownSample(stage_channels[1], act_type=act_type),
            basic_block(stage_channels[1],
                        out_dim=stage_channels[2],
                        expand_ratio=0.25,
                        act_type=act_type,
                        depthwise=depthwise,
                        repconv=repconv,
                        model_size='large',
                        **args)  # P2/4
        )
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=stage_channels[2], act_type=act_type),
            basic_block(in_dim=stage_channels[2],
                        out_dim=stage_channels[3],
                        expand_ratio=0.25,
                        act_type=act_type,
                        depthwise=depthwise,
                        repconv=repconv,
                        model_size='large',
                        **args)  # P3/8
        )

    def forward(self, x):
        stem_out = self.stem(x)
        stage1_out = self.layer_1(stem_out)
        stage2_out = self.layer_2(stage1_out)
        stage3_out = self.layer_3(stage2_out)

        return stem_out, stage1_out, stage2_out, stage3_out

    def __repr__(self) -> str:
        ret = 'BackBone: ' + self.__class__.__name__
        ret += '\n\tin_channels: {}, residual: {}, repconv: {}, stage_channels: {}, act_type: {}'.format(
            self.in_channels, self.residual, self.repconv, self.stage_channels,
            self.act_type)
        return ret


class CustomCSPDNet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 repconv=True,
                 stage_channels=[64, 128, 256, 256],
                 act_type='lrelu',
                 depthwise=False,
                 **args):
        super(CustomCSPDNet, self).__init__()

        self.in_channels = in_channels
        self.repconv = repconv
        self.stage_channels = stage_channels
        self.act_type = act_type

        basic_block = CSPStage

        self.stem = nn.Sequential(
            RepConv(in_channels, 32, identity=False, **args)
            if repconv else Conv(in_channels,
                                 32,
                                 k=3,
                                 p=1,
                                 act_type=act_type,
                                 depthwise=depthwise),
            basic_block(32,
                        stage_channels[0],
                        n=3,
                        depthwise=depthwise,
                        repconv=repconv,
                        **args))
        self.layer_1 = nn.Sequential(
            DownSample(stage_channels[0], act_type=act_type),
            basic_block(stage_channels[0],
                        stage_channels[1],
                        n=3,
                        depthwise=depthwise,
                        repconv=repconv,
                        **args)  # P1/2
        )
        self.layer_2 = nn.Sequential(
            DownSample(stage_channels[1], act_type=act_type),
            basic_block(stage_channels[1],
                        stage_channels[2],
                        n=2,
                        depthwise=depthwise,
                        repconv=repconv,
                        **args)  # P2/4
        )
        self.layer_3 = nn.Sequential(
            DownSample(stage_channels[2], act_type=act_type),
            basic_block(stage_channels[2],
                        stage_channels[3],
                        n=2,
                        depthwise=depthwise,
                        repconv=repconv,
                        **args)  # P3/8
        )

    def forward(self, x):
        stem_out = self.stem(x)
        stage1_out = self.layer_1(stem_out)
        stage2_out = self.layer_2(stage1_out)
        stage3_out = self.layer_3(stage2_out)

        return stem_out, stage1_out, stage2_out, stage3_out

    def __repr__(self) -> str:
        ret = 'BackBone: ' + self.__class__.__name__
        ret += '\n\tin_channels: {}, repconv: {}, stage_channels: {}, act_type: {}'.format(
            self.in_channels, self.repconv, self.stage_channels, self.act_type)
        return ret