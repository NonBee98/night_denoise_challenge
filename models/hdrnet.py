import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import *
from .common_blocks import *


def _guide_map_to_xyz(guide_map: torch.Tensor) -> torch.Tensor:
    """
    Parameters
        guide_map (Tensor) - guide map of shape (B, 1, Hout, Wout)
    Returns
        xyz (Tensor) - xyz coordinates of shape (B, 1, Hout, Wout, 3)
    """
    N, _, H, W = guide_map.shape
    hg, wg = torch.meshgrid(
        [torch.arange(0, H), torch.arange(0, W)], indexing="xy")
    hg = hg.to(dtype=guide_map.dtype, device=guide_map.device)
    wg = wg.to(dtype=guide_map.dtype, device=guide_map.device)

    hg = hg.repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1
    wg = wg.repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1

    guide_map = guide_map.permute(0, 2, 3, 1).contiguous()
    ret = torch.cat([guide_map, hg, wg], dim=-1).unsqueeze(1)
    return ret


def _slicing(guide_map: torch.Tensor,
             bilateral_grid: torch.Tensor) -> torch.Tensor:
    """
    Parameters
        guide_map (Tensor) - guide map of shape (B, 1, Hout, Wout)
        bilateral_grid (Tensor) - bilateral grid (B, N, *Din, Hin, Outin)
    Returns
        coefs (Tensor) - coefficients of shape (B, N, Hout, Wout)
    """
    assert len(guide_map.shape) in (4, 5) and len(bilateral_grid.shape) in (
        4, 5
    ), "guide map and bilateral grid should have same dimension of shape of 4 or 5"

    guide_xyz = _guide_map_to_xyz(guide_map)
    coefs = F.grid_sample(bilateral_grid, guide_xyz, align_corners=False)
    coefs = coefs.squeeze(2)
    return coefs


class Guide(nn.Module):
    '''
    pointwise neural net
    '''
    def __init__(self, cin: int = 4, cmid: int = 16):
        super(Guide, self).__init__()
        self.conv1 = Conv(cin, cmid, k=1, p=0, bias=True)
        self.conv2 = Conv(cmid, 1, k=1, p=0, bias=True)

    def forward(self, x):
        guidemap = self.conv2(self.conv1(x))
        guidemap = torch.tanh(guidemap)

        return guidemap


class LocalPath(nn.Module):
    def __init__(self,
                 cin: int,
                 path_length: int = 2,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.local_features_path = nn.ModuleList()
        for _ in range(path_length):
            self.local_features_path.append(ConvNeXtBlock(cin))

    def forward(self, x):
        for i in range(len(self.local_features_path)):
            x = self.local_features_path[i](x)
        return x


class GlobalPath(nn.Module):
    def __init__(self,
                 cin: int,
                 down_sample_times: int = 2,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.down_samples = nn.ModuleList()
        for _ in range(down_sample_times):
            downsample_layer = nn.Sequential(
                nn.Conv2d(cin, cin, kernel_size=2, stride=2), nn.LeakyReLU())
            self.down_samples.append(downsample_layer)
        dims = [16 * cin, 4 * cin, 2 * cin, cin]
        self.fc_layers = nn.ModuleList()
        for i in range(1, 4):
            fc_layer = nn.Sequential(nn.Linear(dims[i - 1], dims[i]),
                                     nn.LeakyReLU())
            self.fc_layers.append(fc_layer)

    def forward(self, x):
        b, c, h, w = x.shape
        for i in range(len(self.down_samples)):
            x = self.down_samples[i](x)
        x = x.reshape(b, -1)
        for i in range(3):
            x = self.fc_layers[i](x)
        x = x.reshape(b, -1, 1, 1)
        return x


class HDRNet(nn.Module):
    def __init__(self,
                 cin: int,
                 coef_num: int,
                 bin_num: int,
                 input_resolution: tuple[int],
                 dims=[32, 64, 128, 256],
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cin = cin
        self.bin_num = bin_num
        self.coef_num = coef_num
        self.input_resolution = input_resolution

        self.out_dim = cin * coef_num * bin_num

        self.guide = Guide(cin=cin, cmid=16)
        self.backbone = ConvNeXt(cin,
                                 depths=[2, 2, 6, 2],
                                 drop_path_rate=0.1,
                                 dims=dims)
        self.local_features_path = LocalPath(dims[-1])
        self.global_features_path = GlobalPath(dims[-1])
        self.final_conv = nn.Conv2d(dims[-1], self.out_dim, kernel_size=1)

    def forward(self, x):
        x_down = F.interpolate(x,
                               size=self.input_resolution,
                               mode='bilinear',
                               align_corners=False)
        guide_map = self.guide(x)
        features = self.backbone(x_down)
        last_stage_feature = features[-1]
        local_feature = self.local_features_path(last_stage_feature)
        global_feature = self.global_features_path(last_stage_feature)

        fusion_feature = torch.relu(local_feature + global_feature)
        grid = self.final_conv(fusion_feature)
        grid = torch.stack(torch.split(grid, self.cin * self.coef_num, 1), 2)
        coefs = _slicing(guide_map, grid)
        rgb_coefs = torch.split(coefs, self.coef_num, 1)

        return rgb_coefs
