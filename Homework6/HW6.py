"""
    INT13146 - Image Processing (Xu ly anh)
    Homework 5.1
    Student: Ho Duc Hoang (N20DCCN018)
    Class: D20CQCNHT01-N
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# read image salesman from bin file with size 256x256
camera9 = np.fromfile(open('dataset/camera9.bin'), dtype=np.uint8).reshape(256, 256)
camera99 = np.fromfile(open('dataset/camera99.bin'), dtype=np.uint8).reshape(256, 256)


# defined function to use in this homework
def fullScaleContrast(x):
    xMax = np.max(x)
    xMin = np.min(x)
    if xMax - xMin == 0:
        return np.zeros(x.shape)
    Scale_factor = 255.0 / (xMax - xMin)
    return np.round((x - xMin) * Scale_factor)


def show_image(row, col, pos, image, title):
    plt.subplot(row, col, pos)
    plt.imshow(fullScaleContrast(image), cmap='gray')
    plt.title(title, loc='center', wrap=True)
    plt.axis('off')


# implement median filter
def median_filter(_image, _kernel):
    M, N = _image.shape[0], _image.shape[1]
    m, n = _kernel.shape[0], _kernel.shape[1]  # kernel size m x n

    filtered_image = np.zeros((M, N))
    padded_image = np.zeros((M + m // 2 * 2, N + n // 2 * 2))
    padded_image[m // 2: m // 2 + M, n // 2: n // 2 + N] = _image

    for i in range(M):
        for j in range(N):
            filtered_image[i][j] = np.median(padded_image[i:i + m, j:j + n] * _kernel)

    return filtered_image


def dilate(_image, _kernel):
    w, h = _image.shape  # width and height of image input
    kw = len(_kernel)  # size of kernel input
    kw2 = kw // 2

    res = deepcopy(_image)
    for i in range(w - kw + 1):
        for j in range(h - kw + 1):
            W = _image[i:i + kw, j:j + kw]
            res[i + kw2][j + kw2] = np.max(W * _kernel)

    return res


def erode(_image, _kernel):
    w, h = _image.shape  # width and height of image input
    kw = len(_kernel)  # size of kernel input
    kw2 = kw // 2

    res = deepcopy(_image)
    for i in range(w - kw + 1):
        for j in range(h - kw + 1):
            W = _image[i:i + kw, j:j + kw]
            res[i + kw2][j + kw2] = np.min(W * _kernel)

    return res


def morphological(_image, _kernel, mode='opening'):
    if mode == 'opening':
        return dilate(erode(_image, _kernel), _kernel)

    if mode == 'closing':
        return erode(dilate(_image, _kernel), _kernel)

    return None


# create 3 × 3 square structuring element (window)
kernel3 = np.ones((3, 3))

# Median filter on two original images
median_camera9 = median_filter(camera9, kernel3)
median_camera99 = median_filter(camera9, kernel3)

# Morphological opening
opening_camera9 = morphological(camera9, kernel3, 'opening')
opening_camera99 = morphological(camera99, kernel3, 'opening')

# Morphological closing
closing_camera9 = morphological(camera9, kernel3, 'closing')
closing_camera99 = morphological(camera99, kernel3, 'closing')

# show image
show_image(2, 2, 1, camera9, 'camera9.bin')
show_image(2, 2, 2, median_camera9, '3x3 median filter')
show_image(2, 2, 3, opening_camera9, '3x3 morphological opening')
show_image(2, 2, 4, closing_camera9, '3x3 morphological closing')

plt.show()

show_image(2, 2, 1, camera99, 'camera99.bin')
show_image(2, 2, 2, median_camera99, '3x3 median filter')
show_image(2, 2, 3, opening_camera99, '3x3 morphological opening')
show_image(2, 2, 4, closing_camera99, '3x3 morphological closing')

plt.show()
