import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_blocks import *


def _upsample_double(src):
    src = F.interpolate(src, scale_factor=2, mode='nearest')
    return src


class YoloNeck(nn.Module):
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
                 act_type='silu'):
        super(YoloNeck, self).__init__()
        self.fpn = YoloFPN(stage1_channels,
                           stage2_channels,
                           stage3_channels,
                           act_type=act_type)
        fpn_out_channels = self.fpn.get_out_channels()
        self.pan = YoloPAN(*fpn_out_channels)
        self.pan_out_channels = self.pan.get_out_channels()
        self.stage1_rep = RepConv(self.pan_out_channels[0],
                                  self.pan_out_channels[0] * 2)
        self.stage2_rep = RepConv(self.pan_out_channels[1],
                                  self.pan_out_channels[1] * 2)
        self.stage3_rep = RepConv(self.pan_out_channels[2],
                                  self.pan_out_channels[2] * 2)

    def get_out_channels(self):
        return self.pan_out_channels[0] * 2, self.pan_out_channels[
            1] * 2, self.pan_out_channels[2] * 2

    def forward(self, stage1_out, stage2_out, stage3_out):
        fpn_out = self.fpn(stage1_out, stage2_out, stage3_out)
        pan_out = self.pan(*fpn_out)
        out1 = self.stage1_rep(pan_out[0])
        out2 = self.stage2_rep(pan_out[1])
        out3 = self.stage3_rep(pan_out[2])
        return out1, out2, out3


class YoloFPN(nn.Module):
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
                 act_type='silu'):
        super(YoloFPN, self).__init__()

        self.stage3_sppcspc_out_channels = make_divisible(
            stage3_channels // 4, 4)
        self.stage3_sppcspc = SPPCSPC(stage3_channels,
                                      self.stage3_sppcspc_out_channels)
        stage3_cbs_out_channels = make_divisible(
            self.stage3_sppcspc_out_channels // 2, 4)
        self.stage3_cbs = Conv(self.stage3_sppcspc_out_channels,
                               stage3_cbs_out_channels,
                               k=1,
                               s=1,
                               act_type=act_type)

        stage2_cbs_brige_out_channels = make_divisible(stage2_channels // 4, 4)
        self.stage2_cbs_brige = Conv(stage2_channels,
                                     stage2_cbs_brige_out_channels,
                                     k=1,
                                     s=1,
                                     act_type=act_type)
        stage2_elanw_in_channels = stage3_cbs_out_channels + stage2_cbs_brige_out_channels
        self.stage2_elanw_out_channels = make_divisible(
            stage2_elanw_in_channels // 2, 4)
        self.stage2_elanw = ELAN_W_Block(stage2_elanw_in_channels,
                                         self.stage2_elanw_out_channels,
                                         act_type=act_type)
        stage2_cbs_out_channels = make_divisible(
            self.stage2_elanw_out_channels // 2, 4)
        self.stage2_cbs = Conv(self.stage2_elanw_out_channels,
                               stage2_cbs_out_channels,
                               k=1,
                               s=1,
                               act_type=act_type)

        stage1_cbs_brige_out_channels = make_divisible(stage1_channels // 4, 4)
        self.stage1_cbs_brige = Conv(stage1_channels,
                                     stage1_cbs_brige_out_channels,
                                     k=1,
                                     s=1,
                                     act_type=act_type)
        stage1_elanw_in_channels = stage1_cbs_brige_out_channels + stage2_cbs_out_channels
        self.stage1_elanw_out_channels = make_divisible(
            stage1_elanw_in_channels // 2, 4)
        self.stage1_elanw = ELAN_W_Block(stage1_elanw_in_channels,
                                         self.stage1_elanw_out_channels,
                                         act_type=act_type)

    def get_out_channels(self):

        return self.stage1_elanw_out_channels, self.stage2_elanw_out_channels, self.stage3_sppcspc_out_channels

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


class YoloPAN(nn.Module):
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
                 act_type='silu'):
        super(YoloPAN, self).__init__()
        self.stage1_channels = stage1_channels
        self.stage1_mp2 = DownSample2(stage1_channels, act_type=act_type)

        stage2_elanw_in_channels = stage1_channels * 2 + stage2_channels
        self.stage2_elanw_out_channels = make_divisible(
            stage2_elanw_in_channels // 2, 4)
        self.stage2_elanw = ELAN_W_Block(stage2_elanw_in_channels,
                                         self.stage2_elanw_out_channels,
                                         act_type=act_type)
        self.stage2_mp2 = DownSample2(self.stage2_elanw_out_channels,
                                      act_type=act_type)

        stage3_elanw_in_channels = self.stage2_elanw_out_channels * 2 + stage3_channels
        self.stage3_elanw_out_channels = make_divisible(
            stage3_elanw_in_channels // 2, 4)
        self.stage3_elanw = ELAN_W_Block(stage3_elanw_in_channels,
                                         self.stage3_elanw_out_channels,
                                         act_type=act_type)

    def get_out_channels(self):
        return self.stage1_channels, self.stage2_elanw_out_channels, self.stage3_elanw_out_channels

    def forward(self, stage1_out, stage2_out, stage3_out):

        stage1_mp2 = self.stage1_mp2(stage1_out)

        stage2_elanw = self.stage2_elanw(
            torch.cat([stage2_out, stage1_mp2], dim=1))
        stage2_mp2 = self.stage2_mp2(stage2_elanw)

        stage3_elanw = self.stage3_elanw(
            torch.cat([stage3_out, stage2_mp2], dim=1))

        return stage1_out, stage2_elanw, stage3_elanw
