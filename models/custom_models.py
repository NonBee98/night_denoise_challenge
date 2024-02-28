import torch
import torch.nn as nn
from torch.nn import init

from .backbones import *
from .common_blocks import *
from .necks import *


class CUSTOM(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 illum_classes=2,
                 stage_channels=[64, 128, 256, 256],
                 repconv=True,
                 **args):
        super(CUSTOM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.illum_classes = illum_classes
        self.stage_channels = stage_channels
        self.backbone = CustomCSPDNet(in_channels,
                                      stage_channels=stage_channels,
                                      repconv=repconv,
                                      **args)
        self.neck = CustomNeck(*stage_channels,
                               fpn_scale=2,
                               repconv=repconv,
                               **args)
        neck_out_channels = self.neck.get_out_channels()

        self.mask_head = nn.Conv2d(neck_out_channels[0], out_channels, 1)

        self.squeeze = nn.Sequential(Conv(stage_channels[-1], 320, 1, 1, 0),
                                     nn.AdaptiveAvgPool2d((1, 1)),
                                     Conv(320, 320, k=1, bias=True))
        self.gain_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(320, 3 * self.illum_classes + 1))

        self.init_params()

    def set_backbone(self, backbone):
        del self.backbone
        self.backbone = backbone

    def set_neck(self, neck):
        del self.neck
        self.neck = neck

    def forward(self, x):
        stem_out, stage1_out, stage2_out, stage3_out = self.backbone(x)
        neck_out = self.neck(stem_out, stage1_out, stage2_out, stage3_out)

        gain_out = self.squeeze(stage3_out)
        gain_out = gain_out.view(x.size(0), -1)
        gain_out = self.gain_head(gain_out)

        illum_mask = torch.sigmoid(self.mask_head(neck_out[0]))
        illum_gain = gain_out[:, :6]
        illum_class = torch.sigmoid(gain_out[:, 6:])

        return illum_mask, illum_class, illum_gain

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def fuse_params(self):
        for m in self.modules():
            if isinstance(m, (RepConv, Conv)):
                m.fuse_params()

    def quant_convert(self):
        '''
        Prepare for quantization, merge Conv + BN + ReLu
        '''
        for m in self.modules():
            if isinstance(m, Conv):
                m.quant_convert()

    def __repr__(self) -> str:
        ret = 'Model: ' + self.__class__.__name__
        ret += '\n\tin_channels: {}, out_channels: {}, illum_classes: {}, stage_channels: {}'.format(
            self.in_channels, self.out_channels, self.illum_classes,
            self.stage_channels)
        ret += '\n'
        ret += str(self.backbone)
        ret += '\n'
        ret += str(self.neck)
        return ret


class CUSTOM_NO_MASK(nn.Module):

    def __init__(self,
                 in_channels=3,
                 illum_classes=2,
                 stage_channels=[64, 128, 256, 256]):
        super(CUSTOM_NO_MASK, self).__init__()
        self.in_channels = in_channels
        self.illum_classes = illum_classes
        self.stage_channels = stage_channels

        self.backbone = CustomCSPDNet(in_channels,
                                      stage_channels=stage_channels,
                                      repconv=False)

        self.squeeze = nn.Sequential(Conv(stage_channels[-1], 320, 1, 1, 0),
                                     nn.AdaptiveAvgPool2d((1, 1)),
                                     Conv(320, 320, k=1, bias=True))
        self.gain_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(320, 7))

        self.init_params()

    def forward(self, x):
        stem_out, stage1_out, stage2_out, stage3_out = self.backbone(x)

        gain_out = self.squeeze(stage3_out)
        gain_out = gain_out.view(x.size(0), -1)
        gain_out = self.gain_head(gain_out)

        illum_gain = gain_out[:, :6]
        illum_class = torch.sigmoid(gain_out[:, 6:])

        batch_size = illum_class.shape[0]
        illum_mask = torch.ones((batch_size, 1, 64, 64),
                                dtype=illum_class.dtype,
                                device=illum_class.device)

        return illum_mask, illum_class, illum_gain

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def fuse_params(self):
        for m in self.modules():
            if isinstance(m, (RepConv, Conv)):
                m.fuse_params()

    def __repr__(self) -> str:
        ret = 'Model: ' + self.__class__.__name__
        ret += '\n\tin_channels: {}, out_channels: {}, illum_classes: {}, stage_channels: {}'.format(
            self.in_channels, self.out_channels, self.illum_classes,
            self.stage_channels)
        ret += '\n'
        ret += str(self.backbone)
        return ret