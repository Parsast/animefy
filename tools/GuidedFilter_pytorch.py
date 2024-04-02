import torch
import numpy as np
import cv2

def diff_x(input, r):
    assert input.dim() == 4

    left = input[:, :, r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1: -r - 1]

    return torch.cat([left, middle, right], dim=2)


def diff_y(input, r):
    assert input.dim() == 4

    left = input[:, :, :, r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1: -r - 1]

    return torch.cat([left, middle, right], dim=3)


def box_filter(x, r):
    assert x.dim() == 4
    return diff_y(torch.cumsum(diff_x(torch.cumsum(x, dim=2), r), dim=3), r)


def guided_filter(x, y, r, eps=1e-1, nhwc=False):
    """
    @param x: guidance image with value in [0.0 , 1.0]
    @param y: filtering input image with value in [0.0 , 1.0]
    @param r: local window radius : 2, 3, 4, 5
    @param eps: regularization parameter:  0.1**2, 0.2**2, 0.4**2
    @param nhwc: tensor format (NHWC). PyTorch default is NCHW
    @return:  smooth image by guided filter
    """

    assert x.dim() == 4 and y.dim() == 4

    if nhwc:  # Convert to channels-first if necessary
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

    # Shape checks
    x_shape = x.shape
    y_shape = y.shape

    assert x_shape[0] == y_shape[0]
    assert x_shape[2:] == y_shape[2:]
    assert x_shape[2] > 2 * r + 1  # Ensure window size is valid
    assert x.shape[1] == 1 or x.shape[1] == y.shape[1]

    # Calculations (similar to the TensorFlow version)
    N = box_filter(torch.ones((1, 1, x_shape[2], x_shape[3]), device=x.device), r)
    mean_x = box_filter(x, r) / N
    mean_y = box_filter(y, r) / N
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    var_x = box_filter(x * x, r) / N - mean_x * mean_x
    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x
    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b

    if nhwc:
        output = output.permute(0, 2, 3, 1)  # Convert back to NHWC if needed

    return output