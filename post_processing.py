import cv2
import numpy as np
import os
import glob
import tqdm


def pack_raw(im: np.ndarray) -> np.ndarray:
    # pack Bayer image to 4 channels
    im = np.expand_dims(im, axis=-1)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[0:H:2, 0:W:2, :], im[0:H:2, 1:W:2, :],
                    im[1:H:2, 1:W:2, :], im[1:H:2, 0:W:2, :]),
                   axis=-1)
    return out


def simple_demosaic(img, cfa_pattern=[0, 1, 1, 2]):
    raw_colors = np.asarray(cfa_pattern).reshape((2, 2))
    demosaiced_image = np.zeros((img.shape[0] // 2, img.shape[1] // 2, 3))
    for i in range(2):
        for j in range(2):
            ch = raw_colors[i, j]
            if ch == 1:
                demosaiced_image[:, :, ch] += img[i::2, j::2] / 2
            else:
                demosaiced_image[:, :, ch] = img[i::2, j::2]
    return demosaiced_image


def grey_world(img):
    # Calculate the average illumination of the image
    shape = img.shape
    img = img.reshape((-1, 3))
    avg_illumination = np.mean(img, axis=0, keepdims=True)
    avg_illumination /= (avg_illumination[..., 1] + 1e-6)
    img /= avg_illumination
    img = img.reshape(shape)
    return img


ccm = np.array([[1.06835938, -0.29882812, -0.14257812],
                [-0.43164062, 1.35546875, 0.05078125],
                [-0.1015625, 0.24414062, 0.5859375]])


def main():
    raw_dir = './test_outputs/'
    out_dir = './test_outputs_awb/'
    os.makedirs(out_dir, exist_ok=True)
    img_list = glob.glob(os.path.join(raw_dir, '*.png'))
    for img_path in tqdm.tqdm(img_list):
        basename = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        img = simple_demosaic(img)
        img = grey_world(img)
        img = img ** (1/2.2)
        img *= 65535.0
        img = np.clip(img, 0, 65535)
        cv2.imwrite(os.path.join(out_dir, basename), img.astype(np.uint16))


if __name__ == '__main__':
    main()
