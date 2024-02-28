import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_blocks import *


def _upsample_double(src):
    src = F.interpolate(src, scale_factor=2, mode='nearest')
    return src


def _upsample_like(src, tar):

    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')

    return src


class CustomNeck(nn.Module):
    """
    Input: 64 * 64
    Stage 1: 32 * 32
    Stage 2: 16 * 16
    Stage 3: 8 * 8
    """

    def __init__(self,
                 stage0_channels,
                 stage1_channels,
                 stage2_channels,
                 stage3_channels,
                 repconv=True,
                 fpn_scale=2,
                 **args):
        super(CustomNeck, self).__init__()
        self.repconv = repconv
        self.fpn_scale = fpn_scale

        self.fpn = CustomFPN4Stages(stage0_channels,
                                    stage1_channels,
                                    stage2_channels,
                                    stage3_channels,
                                    repconv=repconv,
                                    fpn_scale=fpn_scale,
                                    **args)
        self.out_channels = self.fpn.get_out_channels()

    def forward(self, stage0_out, stage1_out, stage2_out, stage3_out):
        out = self.fpn(stage0_out, stage1_out, stage2_out, stage3_out)
        return out

    def get_out_channels(self):
        return self.out_channels

    def __repr__(self) -> str:
        ret = 'Neck: {} with {}'.format(self.__class__.__name__,
                                        self.fpn.__class__.__name__)
        ret += '\n\trepconv: {}'.format(self.repconv)
        return ret


class CustomFPN4Stages(nn.Module):
    """
    Input: 64 * 64
    Stage 0: 64 * 64
    Stage 1: 32 * 32
    Stage 2: 16 * 16
    Stage 3: 8 * 8
    """

    def __init__(self,
                 stage0_channels,
                 stage1_channels,
                 stage2_channels,
                 stage3_channels,
                 repconv=True,
                 fpn_scale=2,
                 **args):
        super(CustomFPN4Stages, self).__init__()

        self.stage3_out_channels = make_divisible(
            stage3_channels // (fpn_scale * 1), 4)
        self.stage3_brige = Conv(stage3_channels,
                                 self.stage3_out_channels,
                                 k=1,
                                 s=1)

        stage3_cbs_out_channels = make_divisible(self.stage3_out_channels // 2,
                                                 4)
        self.stage3_cbs = Conv(self.stage3_out_channels,
                               stage3_cbs_out_channels,
                               k=3,
                               s=1)
        self.stage3_upsample = UpSample(stage3_cbs_out_channels)

        stage2_cbs_brige_out_channels = make_divisible(
            stage2_channels // fpn_scale, 4)
        self.stage2_cbs_brige = Conv(
            stage2_channels,
            stage2_cbs_brige_out_channels,
            k=1,
            s=1,
        )
        stage2_elanw_in_channels = stage3_cbs_out_channels + stage2_cbs_brige_out_channels
        self.stage2_out_channels = make_divisible(
            stage2_elanw_in_channels // 2, 4)
        self.stage2_elanw = RepConv(
            stage2_elanw_in_channels, self.stage2_out_channels, **
            args) if repconv else Conv(
                stage2_elanw_in_channels, self.stage2_out_channels, k=3, s=1)
        stage2_cbs_out_channels = make_divisible(self.stage2_out_channels // 2,
                                                 4)
        self.stage2_cbs = Conv(self.stage2_out_channels,
                               stage2_cbs_out_channels,
                               k=3,
                               s=1)
        self.stage2_upsample = UpSample(stage2_cbs_out_channels)

        stage1_cbs_brige_out_channels = make_divisible(
            stage1_channels // fpn_scale, 4)
        self.stage1_cbs_brige = Conv(stage1_channels,
                                     stage1_cbs_brige_out_channels,
                                     k=1,
                                     s=1)
        stage1_elanw_in_channels = stage1_cbs_brige_out_channels + stage2_cbs_out_channels
        self.stage1_out_channels = make_divisible(
            stage1_elanw_in_channels // 2, 4)
        self.stage1_elanw = RepConv(
            stage1_elanw_in_channels, self.stage1_out_channels, **
            args) if repconv else Conv(
                stage1_elanw_in_channels, self.stage1_out_channels, k=3, s=1)
        stage1_cbs_out_channels = make_divisible(self.stage1_out_channels // 2,
                                                 4)
        self.stage1_cbs = Conv(self.stage1_out_channels,
                               stage1_cbs_out_channels,
                               k=3,
                               s=1)
        self.stage1_upsample = UpSample(stage1_cbs_out_channels)

        stage0_cbs_brige_out_channels = make_divisible(
            stage0_channels // fpn_scale, 4)
        self.stage0_cbs_brige = Conv(stage0_channels,
                                     stage0_cbs_brige_out_channels,
                                     k=1,
                                     s=1)
        stage0_elanw_in_channels = stage0_cbs_brige_out_channels + stage1_cbs_out_channels
        self.stage0_out_channels = make_divisible(
            stage0_elanw_in_channels // 2, 4)
        self.stage0_elanw = RepConv(
            stage0_elanw_in_channels, self.stage0_out_channels, **
            args) if repconv else Conv(
                stage0_elanw_in_channels, self.stage0_out_channels, k=3, s=1)

    def get_out_channels(self):

        return self.stage0_out_channels, self.stage1_out_channels, self.stage2_out_channels, self.stage3_out_channels

    def forward(self, stage0_out, stage1_out, stage2_out, stage3_out):
        stage3_brige = self.stage3_brige(stage3_out)
        stage3_cbs = self.stage3_cbs(stage3_brige)
        stage3_cbs = self.stage3_upsample(stage3_cbs)

        stage2_brige = self.stage2_cbs_brige(stage2_out)
        stage2_elanw = self.stage2_elanw(
            torch.cat([stage2_brige, stage3_cbs], dim=1))
        stage2_cbs = self.stage2_cbs(stage2_elanw)
        stage2_cbs = self.stage2_upsample(stage2_cbs)

        stage1_brige = self.stage1_cbs_brige(stage1_out)
        stage1_elanw = self.stage1_elanw(
            torch.cat([stage1_brige, stage2_cbs], dim=1))
        stage1_cbs = self.stage1_cbs(stage1_elanw)
        stage1_cbs = self.stage1_upsample(stage1_cbs)

        stage0_brige = self.stage0_cbs_brige(stage0_out)
        stage0_elanw = self.stage0_elanw(
            torch.cat([stage0_brige, stage1_cbs], dim=1))

        return stage0_elanw, stage1_elanw, stage2_elanw, stage3_brige


class CustomFPN(nn.Module):
    """
    Input: 64 * 64
    Stage 1: 32 * 32
    Stage 2: 16 * 16
    Stage 3: 8 * 8
    """

    def __init__(self,
                 stage1_channels,
                 stage2_channels,
                 stage3_channels,
                 fpn_scale=4):
        super(CustomFPN, self).__init__()

        self.stage3_out_channels = make_divisible(stage3_channels // fpn_scale,
                                                  4)
        self.stage3_sppcspc = Conv(stage3_channels,
                                   self.stage3_out_channels,
                                   k=1,
                                   s=1)

        stage3_cbs_out_channels = make_divisible(self.stage3_out_channels // 2,
                                                 4)
        self.stage3_cbs = Conv(self.stage3_out_channels,
                               stage3_cbs_out_channels,
                               k=1,
                               s=1)

        stage2_cbs_brige_out_channels = make_divisible(
            stage2_channels // fpn_scale, 4)
        self.stage2_cbs_brige = Conv(
            stage2_channels,
            stage2_cbs_brige_out_channels,
            k=1,
            s=1,
        )
        stage2_elanw_in_channels = stage3_cbs_out_channels + stage2_cbs_brige_out_channels
        self.stage2_out_channels = make_divisible(
            stage2_elanw_in_channels // 2, 4)
        self.stage2_elanw = nn.Conv2d(stage2_elanw_in_channels,
                                      self.stage2_out_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        stage2_cbs_out_channels = make_divisible(self.stage2_out_channels // 2,
                                                 4)
        self.stage2_cbs = nn.Conv2d(self.stage2_out_channels,
                                    stage2_cbs_out_channels,
                                    kernel_size=1,
                                    stride=1)

        stage1_cbs_brige_out_channels = make_divisible(
            stage1_channels // fpn_scale, 4)
        self.stage1_cbs_brige = nn.Conv2d(stage1_channels,
                                          stage1_cbs_brige_out_channels,
                                          kernel_size=1,
                                          stride=1)
        stage1_elanw_in_channels = stage1_cbs_brige_out_channels + stage2_cbs_out_channels
        self.stage1_out_channels = make_divisible(
            stage1_elanw_in_channels // 2, 4)
        self.stage1_elanw = nn.Conv2d(stage1_elanw_in_channels,
                                      self.stage1_out_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)

    def get_out_channels(self):

        return self.stage1_out_channels, self.stage2_out_channels, self.stage3_out_channels

    def forward(self, stage1_out, stage2_out, stage3_out):
        stage3_sppcspc = self.stage3_sppcspc(stage3_out)
        stage3_cbs = self.stage3_cbs(stage3_sppcspc)
        stage3_cbs = _upsample_double(stage3_cbs)

        stage2_brige = self.stage2_cbs_brige(stage2_out)
        stage2_elanw = self.stage2_elanw(
            torch.cat([stage2_brige, stage3_cbs], dim=1))
        stage2_cbs = self.stage2_cbs(stage2_elanw)
        stage2_cbs = _upsample_double(stage2_cbs)

        stage1_brige = self.stage1_cbs_brige(stage1_out)
        stage1_elanw = self.stage1_elanw(
            torch.cat([stage1_brige, stage2_cbs], dim=1))

        return stage1_elanw, stage2_elanw, stage3_sppcspc


class CustomPAN(nn.Module):
    """
    Input: 64 * 64
    Stage 1: 32 * 32
    Stage 2: 16 * 16
    Stage 3: 8 * 8
    """

    def __init__(self,
                 stage1_channels,
                 stage2_channels,
                 stage3_channels,
                 act_type='lrelu'):
        super(CustomPAN, self).__init__()
        self.stage1_channels = stage1_channels
        self.stage1_mp2 = DownSample2(stage1_channels, act_type=act_type)

        stage2_elanw_in_channels = stage1_channels * 2 + stage2_channels
        self.stage2_out_channels = make_divisible(
            stage2_elanw_in_channels // 2, 4)
        self.stage2_elanw = nn.Conv2d(stage2_elanw_in_channels,
                                      self.stage2_out_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.stage2_mp2 = DownSample2(self.stage2_out_channels,
                                      act_type=act_type)

        stage3_elanw_in_channels = self.stage2_out_channels * 2 + stage3_channels
        self.stage3_elanw_out_channels = make_divisible(
            stage3_elanw_in_channels // 2, 4)
        self.stage3_elanw = nn.Conv2d(stage3_elanw_in_channels,
                                      self.stage3_elanw_out_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)

    def get_out_channels(self):
        return self.stage1_channels, self.stage2_out_channels, self.stage3_elanw_out_channels

    def forward(self, stage1_out, stage2_out, stage3_out):

        stage1_mp2 = self.stage1_mp2(stage1_out)

        stage2_elanw = self.stage2_elanw(
            torch.cat([stage2_out, stage1_mp2], dim=1))
        stage2_mp2 = self.stage2_mp2(stage2_elanw)

        stage3_elanw = self.stage3_elanw(
            torch.cat([stage3_out, stage2_mp2], dim=1))

        return stage1_out, stage2_elanw, stage3_elanw
