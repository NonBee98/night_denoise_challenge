import torch
import torch.nn as nn
import tqdm
from torch import Tensor, nn, optim, utils
from torchvision.transforms.functional import equalize

from data_loader import *
from model import *


def compute_psnr(img1: Tensor, img2: Tensor, saturation=1) -> float:
    mse = nn.functional.mse_loss(img1 * saturation, img2 * saturation)
    return 10 * torch.log10(saturation**2 / mse)


class ModelEncaupsulation(nn.Module):
    def __init__(self, model):
        super(ModelEncaupsulation, self).__init__()
        self.model = model()

    def forward(self, x):
        return self.model(x)


def compute_exposure_ratio_by_hist(img: Tensor,
                                   target_luminance: float) -> float:
    """
    Compute the exposure ratio by histogram equalization.
    Args:
        img: input image with shape (h, w, c), range [0, 1]
        target_luminance: target luminance
    Returns:
        exposure_ratio: the exposure ratio to achieve the target luminance
    """
    img = img.permute(1, 2, 0)
    img = depack_raw(img)
    h = img.shape[0]
    w = img.shape[1]
    target_img_size = 256
    h_step = h // target_img_size
    w_step = w // target_img_size
    down_sampled_image = img[::h_step, ::w_step] * 255
    down_sampled_image = down_sampled_image.unsqueeze(0)
    down_sampled_image_hist_equalized = equalize(
        down_sampled_image.to(torch.uint8)).float()
    mean = down_sampled_image_hist_equalized.mean() / 255
    print(img.mean(), mean, target_luminance)
    ret = target_luminance / mean

    return ret


def main():
    model = ModelEncaupsulation(UNetSeeInDark)
    parameters = torch.load('./checkpoints/sid_best.ckpt')
    model.bfloat16()
    model.load_state_dict(parameters['state_dict'])
    model.eval()
    model = model.cuda()
    luminance_level = 50
    target_luminance = luminance_level / 255
    bit_depth = 16
    save_type = np.uint8 if bit_depth == 8 else np.uint16

    save_dir = './test_outputs_{}_resized'.format(luminance_level)
    nosiy_save_dir = './test_inputs'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(nosiy_save_dir, exist_ok=True)

    patchify = False
    test_dataset = TestDataset('./test_image/', patchify=patchify, down_sample_rate=4)
    len_data = len(test_dataset)
    for i in tqdm.tqdm(range(len_data)):
        data, shape = test_dataset[i]
        data_mean_lluminancce = data.mean()
        exposure_ratio = target_luminance / data_mean_lluminancce
        img_h = shape[0]
        img_w = shape[1]
        file_name = test_dataset.get_basename(i)
        save_name = os.path.join(save_dir, file_name)
        inputs = data * exposure_ratio
        inputs = torch.clamp(inputs, 0, 1)
        inputs = inputs.cuda().bfloat16()
        with torch.no_grad():
            if not patchify:
                inputs = inputs.unsqueeze(0)
            outputs = model(inputs)
            outputs *= (2**bit_depth - 1)
            outputs = outputs.float()
            outputs = outputs.permute(0, 2, 3, 1).cpu()
            outputs = torch.clamp(outputs, 0, 2**bit_depth - 1)
            if patchify:
                outputs = depatchify(outputs, img_h // 2, img_w // 2)
            else:
                outputs = outputs.squeeze()
            outputs = depack_raw(outputs).numpy().astype(save_type)
            cv2.imwrite(save_name, outputs)

        save_name = os.path.join(nosiy_save_dir, file_name)
        if not os.path.exists(save_name):
            with torch.no_grad():
                outputs = inputs
                outputs *= (2**bit_depth - 1)
                outputs = outputs.permute(0, 2, 3, 1).cpu()
                outputs = outputs.float()
                outputs = torch.clamp(outputs, 0, 2**bit_depth - 1)
                if patchify:
                    outputs = depatchify(outputs, img_h // 2, img_w // 2)
                else:
                    outputs = outputs.squeeze()
                outputs = depack_raw(outputs).numpy().astype(save_type)
                cv2.imwrite(save_name, outputs)


if __name__ == '__main__':
    main()
