import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import math

class Erosion2d(nn.Layer):
    """
    Erosion2d
    """
    def __init__(self, m=1):
        super(Erosion2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=1e9)
        channel = nn.functional.unfold(x_pad, 2 * self.m + 1, strides=1, paddings=0).reshape([batch_size, c, -1, h, w])
        result = paddle.min(channel, axis=2)
        return result

class Dilation2d(nn.Layer):
    """
    Dilation2d
    """
    def __init__(self, m=1):
        super(Dilation2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=-1e9)
        channel = nn.functional.unfold(x_pad, 2 * self.m + 1, strides=1, paddings=0).reshape([batch_size, c, -1, h, w])
        result = paddle.max(channel, axis=2)
        return result

def param2stroke(param, H, W, meta_brushes):
    """
    param2stroke
    """
    b = param.shape[0]
    param_list = paddle.split(param, 8, axis=1)
    x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
    sin_theta = paddle.sin(math.pi * theta)
    cos_theta = paddle.cos(math.pi * theta)
    index = paddle.full((b,), -1, dtype='int64').numpy()

    index[(h > w).numpy()] = 0
    index[(h <= w).numpy()] = 1
    meta_brushes_resize = F.interpolate(meta_brushes, (H, W)).numpy()
    brush = paddle.to_tensor(meta_brushes_resize[index])

    warp_00 = cos_theta / w
    warp_01 = sin_theta * H / (W * w)
    warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
    warp_10 = -sin_theta * W / (H * h)
    warp_11 = cos_theta / h
    warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
    warp_0 = paddle.stack([warp_00, warp_01, warp_02], axis=1)
    warp_1 = paddle.stack([warp_10, warp_11, warp_12], axis=1)
    warp = paddle.stack([warp_0, warp_1], axis=1)
    grid = nn.functional.affine_grid(warp, [b, 3, H, W]) # paddle和torch默认值是反过来的
    brush = nn.functional.grid_sample(brush, grid) 
    return brush


def read_img(img_path, img_type='RGB', h=None, w=None):
    """
    read img
    """
    img = Image.open(img_path).convert(img_type)
    if h is not None and w is not None:
        img = img.resize((w, h), resample=Image.NEAREST)
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1))
    img = paddle.to_tensor(img).unsqueeze(0).astype('float32') / 255.
    return img

def preprocess(img, w=512, h=512):
    image = cv2.resize(img, (w, h), cv2.INTER_NEAREST)
    image = image.transpose((2, 0, 1))
    image = paddle.to_tensor(image).unsqueeze(0).astype('float32') / 255.
    return image

def pad(img, H, W):
    b, c, h, w = img.shape
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    expand_img = nn.functional.pad(img, [pad_w, pad_w + remainder_w,
                                         pad_h, pad_h + remainder_h])
    return expand_img



