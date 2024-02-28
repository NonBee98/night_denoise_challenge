from __future__ import division, print_function

import math
import random
from typing import Dict, List, Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms


def _compute_zero_padding(kernel_size: tuple):
    r"""Utility function that computes zero padding tuple."""
    computed = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]


def _get_binary_kernel2d(window_size):
    r"""Create a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def median_blur(input: torch.Tensor, kernel_size: tuple) -> torch.Tensor:
    r"""Blur an image using the median filter.

    .. image:: _static/img/median_blur.png

    Args:
        input: the input image with shape :math:`(C,H,W)`.
        kernel_size: the blurring kernel size.

    Returns:
        the blurred input tensor with shape :math:`(C,H,W)`.
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 3:
        raise ValueError(
            f"Invalid input shape, we expect CxHxW. Got: {input.shape}")

    padding = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: torch.Tensor = _get_binary_kernel2d(kernel_size).to(input)
    c, h, w = input.shape

    # map the local window to single vector
    features: torch.Tensor = F.conv2d(input.reshape(c, 1, h, w),
                                      kernel,
                                      padding=padding,
                                      stride=1)
    features = features.view(c, -1, h, w)

    median: torch.Tensor = torch.median(features, dim=1)[0]

    return median


def expand_all_data_dim(data: dict, dim=0):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            v = torch.unsqueeze(v, dim=dim)
            data[k] = v
    return data


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (dict of ``Transform`` objects): dict of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datas: dict):
        for t in self.transforms:
            datas = t(datas)
        return datas

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class PermuteToCHW(object):

    def __init__(
            self,
            apply_on=['inputs', 'labels']) -> None:
        self.apply_on = set(apply_on)

    def __call__(self, datas: dict):
        for k, v in datas.items():
            if k in self.apply_on:
                if len(v.shape) < 4:
                    v = torch.unsqueeze(v, dim=0)
                v = v.permute((0, 3, 1, 2))
                datas[k] = v
        return datas


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, datas: dict):
        for k, v in enumerate(datas):
            if v is not None:
                v = torch.from_numpy(v)
                datas[k] = v
        return datas


class RandomCrop(object):

    def __init__(self,
                 size,
                 ratio=0.5,
                 apply_on=['inputs', 'labels', 'wbImgs']):
        self.crop_size = size
        self.ratio = ratio
        self.apply_on = set(apply_on)

    def __call__(self, datas: dict):
        # Random crop
        if random.random() < self.ratio:
            i, j, h, w = 0, 0, 0, 0
            for k in self.apply_on:
                data = datas[k]
                i, j, h, w = transforms.RandomCrop.get_params(
                    data, output_size=(self.crop_size, self.crop_size))
                break
            for k, v in datas.items():
                if k in self.apply_on and v is not None:
                    v = TF.crop(v, i, j, h, w)
                    datas[k] = v
        return datas


class RandomFlip(object):

    def __init__(self,
                 ratio=0.5,
                 vertical=True,
                 horizontal=True,
                 apply_on=['inputs', 'labels', 'wbImgs']):
        self.ratio = ratio
        self.vertical = vertical
        self.horizontal = horizontal
        self.apply_on = set(apply_on)

    def __call__(self, datas: dict):
        if self.horizontal and random.random() < self.ratio:
            for k, v in datas.items():
                if k in self.apply_on and v is not None:
                    datas[k] = TF.hflip(v)

        # Random vertical flipping
        if self.vertical and random.random() < self.ratio:
            for k, v in datas.items():
                if k in self.apply_on and v is not None:
                    datas[k] = TF.vflip(v)

        return datas


class Random90Rotate(object):

    def __init__(self,
                 ratio=0.5,
                 apply_on=['inputs', 'labels']):
        self.ratio = ratio
        self.apply_on = set(apply_on)

    def __call__(self, datas: dict):
        if random.random() < self.ratio:
            angle = random.choice([1, 2, 3, 4])  # 0:0 1:90 2:180, 3:270
            for k, v in datas.items():
                if k in self.apply_on and v is not None:
                    datas[k] = torch.rot90(datas[k], angle, [2, 3])
        return datas


class Resize(object):

    def __init__(self,
                 size,
                 apply_on=['inputs', 'labels']):
        assert isinstance(size, (int, tuple, dict))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.apply_on = set(apply_on)

    def __call__(self, datas: dict):
        for k, v in datas.items():
            if k in self.apply_on and v is not None:
                datas[k] = TF.resize(datas[k], self.size)
        return datas


class RandomResize(object):

    def __init__(self,
                 default_size,
                 target_size,
                 ratio=0.2,
                 apply_on=['inputs', 'labels']):
        self.ratio = ratio
        assert isinstance(default_size, (int, tuple, dict))
        if isinstance(default_size, int):
            self.default_size = (default_size, default_size)
        else:
            self.default_size = default_size

        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = default_size
        self.apply_on = set(apply_on)

    def __call__(self, datas: dict):
        size = self.target_size if random.random(
        ) < self.ratio else self.default_size

        for k, v in datas.items():
            if k in self.apply_on and v is not None:
                datas[k] = TF.resize(datas[k], size)
        return datas


class RandomCropResize(object):

    def __init__(self,
                 ratio=0.5,
                 apply_on=['inputs', 'labels']):
        self.ratio = ratio
        self.apply_on = set(apply_on)

    def __call__(self, datas: dict):
        if random.random() < self.ratio:
            shape = (0, 0)
            i, j, h, w = 0, 0, 0, 0
            scale_ratio = random.uniform(0.8, 1)

            for k in self.apply_on:
                data = datas[k]
                if data is not None:
                    shape = (data.shape[-2], data.shape[-1])
                    crop_size = (int(shape[0] * scale_ratio),
                                 int(shape[1] * scale_ratio))
                    i, j, h, w = transforms.RandomCrop.get_params(
                        data, output_size=crop_size)
                    break

            for k, v in datas.items():
                if k in self.apply_on and v is not None:
                    datas[k] = TF.crop(datas[k], i, j, h, w)
                    datas[k] = TF.resize(datas[k], shape)
        return datas


class RandomRotate(object):

    def __init__(self,
                 ratio=0.5,
                 ranges=30,
                 apply_on=['inputs', 'labels']):
        self.ratio = ratio
        self.ranges = ranges
        self.apply_on = set(apply_on)

    def __call__(self, datas: dict):
        if random.random() < self.ratio:
            angle = random.randint(-self.ranges, self.ranges)
            for k, v in datas.items():
                if k in self.apply_on and v is not None:
                    datas[k] = kornia.geometry.rotate(
                        datas[k],
                        torch.tensor(angle,
                                     dtype=datas[k].dtype,
                                     device=datas[k].device),
                        padding_mode='reflection')
        return datas
