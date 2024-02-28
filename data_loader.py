import glob
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import math


def pack_raw(im: torch.Tensor) -> torch.Tensor:
    # pack Bayer image to 4 channels
    im = torch.unsqueeze(im, dim=-1)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = torch.cat((im[0:H:2, 0:W:2, :], im[0:H:2, 1:W:2, :],
                     im[1:H:2, 1:W:2, :], im[1:H:2, 0:W:2, :]),
                    dim=-1)
    return out


def depack_raw(im: torch.Tensor) -> torch.Tensor:
    # unpack 4 channels to Bayer image
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    output = torch.zeros((H * 2, W * 2))
    img_shape = output.shape
    H = img_shape[0]
    W = img_shape[1]

    output[0:H:2, 0:W:2] = im[:, :, 0]
    output[0:H:2, 1:W:2] = im[:, :, 1]
    output[1:H:2, 1:W:2] = im[:, :, 2]
    output[1:H:2, 0:W:2] = im[:, :, 3]

    return output


def patchify(im: torch.Tensor, patch_size: int = 512) -> torch.Tensor:

    # split image into patches
    h, w, c = im.shape
    num_patches_h = math.ceil(h / patch_size)
    num_patches_w = math.ceil(w / patch_size)
    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            col = i * patch_size
            row = j * patch_size
            if col + patch_size > h:
                col = h - patch_size
            if row + patch_size > w:
                row = w - patch_size
            patch = im[col:col + patch_size, row:row + patch_size, :]
            patches.append(patch)
    patches = torch.stack(patches, dim=0)
    return patches


def depatchify(patches: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # merge patches into image
    patch_h = patches[0].shape[0]
    patch_w = patches[0].shape[1]
    h_patches = math.ceil(h / patch_h)
    w_patches = math.ceil(w / patch_w)
    im = torch.zeros((h, w, 4))
    for i in range(h_patches):
        for j in range(w_patches):
            row = i * patch_h
            col = j * patch_w
            if row + patch_h > h:
                row = h - patch_h
            if col + patch_w > w:
                col = w - patch_w
            im[row:row + patch_h,
               col:col + patch_w, :] = patches[i * w_patches + j]
    return im


def generate_poisson_(y, k=1):
    y = torch.poisson(y / k) * k
    return y


def generate_read_noise(shape, noise_type, scale, loc=0):
    noise_type = noise_type.lower()
    if noise_type == 'norm':
        read = torch.FloatTensor(shape).normal_(loc, scale)
    else:
        raise NotImplementedError('Read noise type error.')
    return read


def sample_params_from_mate_40_pro():
    wp = 4095
    bl = 256
    params = {}
    params['Kmin'] = 2.32350645e-06
    params['Kmax'] = 0.012031522716
    params['sigGsk'] = 1.87026223
    params['sigGsb'] = 0.68808067
    params['sigGssig'] = 0.02921

    log_K = np.random.uniform(low=params['Kmin'], high=params['Kmax'])
    K = np.exp(log_K)
    mu_Gs = params['sigGsk'] * log_K + params['sigGsb']

    log_sigGs = np.random.normal(loc=mu_Gs, scale=params['sigGssig'])
    sigGs = np.exp(log_sigGs)
    ratio = np.random.uniform(low=1, high=200)

    return {'K': K, 'sigGs': sigGs, 'ratio': ratio, 'wp': wp, 'bl': bl}


def add_noise(img: torch.Tensor):
    img = torch.clone(img)
    params = sample_params_from_mate_40_pro()
    k = params['K']
    sigGs = params['sigGs']
    ratio = params['ratio']
    wp = params['wp']
    bl = params['bl']
    data_range = wp - bl
    img = img * data_range / ratio
    shot_noise = generate_poisson_(img, k)
    read_noise = generate_read_noise(img.shape, 'norm', sigGs)
    noisy_q = np.random.uniform(low=-0.5, high=0.5, size=img.shape)

    noisy = shot_noise + read_noise + noisy_q
    noisy = noisy * ratio / data_range
    noisy = noisy.clamp(0, 1)
    return noisy


class NoiseDataset(Dataset):
    def __init__(self, folder_path: str, duplicate=5):
        scene_list = os.listdir(os.path.join(folder_path, 'gt'))
        scene_list = list(
            filter(lambda x: os.path.isdir(os.path.join(folder_path, 'gt', x)),
                   scene_list))
        scene_list.sort()
        self.label_list = []
        for scene_name in scene_list:
            label_list = glob.glob(
                os.path.join(folder_path, 'gt', scene_name, "*.pth"))
            self.label_list += label_list
        self.label_list = self.label_list * duplicate

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        ret = {}
        clean_image = torch.load(self.label_list[idx]).float()
        noisy_image = add_noise(clean_image)

        ret["inputs"] = noisy_image.permute(2, 0, 1).float()
        ret["labels"] = clean_image.permute(2, 0, 1).float()
        return ret


class TestDataset(Dataset):
    def __init__(self, dir_path: str, patchify=True) -> None:
        super().__init__()
        support_ext = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        self.image_list = glob.glob(os.path.join(dir_path, "*"))
        self.image_list = list(
            filter(lambda x: os.path.splitext(x)[-1].lower() in support_ext,
                   self.image_list))
        self.basename_list = [os.path.basename(x) for x in self.image_list]
        self.patchify = patchify

    def __len__(self):
        return len(self.image_list)

    def get_basename(self, idx):
        return self.basename_list[idx]

    def __getitem__(self, idx):
        img = cv2.imread(self.image_list[idx],
                         cv2.IMREAD_UNCHANGED).astype(np.float32)
        shape = img.shape
        img /= 4096
        img = torch.from_numpy(img)
        img = pack_raw(img)
        if self.patchify:
            img = patchify(img, 512)
            img = img.permute(0, 3, 1, 2)
        else:
            img = img.permute(2, 0, 1)
        return img, shape
