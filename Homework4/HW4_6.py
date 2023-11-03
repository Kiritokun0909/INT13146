""""
    INT 13146 - Xu Ly Anh
    Homework 4.6
    Ho Duc Hoang - N20DCCN018 - D20CQCHT01-N
"""
import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

SIZE = 256  # define size of image

# define path to dataset
path = 'dataset/'
images_file_name = [
    'camera'
    , 'salesman'
    , 'head'
    , 'eyeR']
num_of_image = len(images_file_name)

# define num of row and column use to draw
ROWS, COLS = 3, 2


def show_image(rows, cols, pos, title, img):
    plt.subplot(rows, cols, pos)
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')


def fullScaleContrast(img, size):
    res = np.zeros((size, size))
    _min = np.min(img)
    _max = np.max(img)

    scaleFact = 0
    if (_max - _min) != 0:
        scaleFact = 255.0 / (_max - _min)

    for u in range(size):
        for v in range(size):
            res[u][v] = round(scaleFact * (img[u][v] - _min))

    return res


# read image from file to arrays
images = [np.fromfile(open(path + img + '.bin'), dtype=np.uint8).reshape(SIZE, SIZE) for img in images_file_name]


# show the original image,
# real part of the centered DFT,
# the imaginary part of the centered DFT
# the centered DFT log-magnitude spectrum,
# and the phase of centered DFT as 8 bpp images
# with full-scale contrast


for i in range(num_of_image):
    X = images[i]  # original image
    show_image(ROWS, COLS, 1, images_file_name[i], X)

    # Compute the DFT I5
    DFT_X = np.fft.fft2(X)
    DFT_X = np.fft.fftshift(DFT_X)  # center it
    XtildeR = np.real(DFT_X)  # real part
    XtildeI = np.imag(DFT_X)  # imaginary part

    # the centered DFT log-magnitude spectrum
    # the phase of centered DFT
    XtildeMag = np.log(np.abs(DFT_X) + 1)
    XtildePhase = np.angle(DFT_X)

    # full scale contrast
    XtildeR = fullScaleContrast(XtildeR, SIZE)
    XtildeI = fullScaleContrast(XtildeI, SIZE)
    XtildeMag = fullScaleContrast(XtildeMag, SIZE)
    XtildePhase = fullScaleContrast(XtildePhase, SIZE)

    # show image
    show_image(ROWS, COLS, 3, 'Re[DFT(' + images_file_name[i] + ')]', XtildeR)
    show_image(ROWS, COLS, 4, 'Im[DFT(' + images_file_name[i] + ')]', XtildeI)
    show_image(ROWS, COLS, 5, 'Log-Magnitude Spectrum', XtildeMag)
    show_image(ROWS, COLS, 6, 'arg[DFT(' + images_file_name[i] + ')]', XtildePhase)

    plt.tight_layout()
    plt.show()
