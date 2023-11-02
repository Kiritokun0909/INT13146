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

# define size of image
WIDTH, HEIGHT = 256, 256

# define path to dataset
dataset = 'dataset/'
images_file_name = [
    'camera'
    , 'salesman'
    , 'head'
    , 'eyeR']
num_of_image = len(images_file_name)


# define num of row and column use to draw
ROWS = 3
COLS = 2


def show_image(rows, cols, pos, title, img):
    plt.subplot(rows, cols, pos)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img, cmap='gray')


def fullScaleContrast(img):
    res = np.zeros((WIDTH, HEIGHT))
    _min = np.min(img)
    _max = np.max(img)

    scaleFact = 0
    if (_max - _min) != 0:
        scaleFact = 255.0 / (_max - _min)

    for i in range(WIDTH):
        for j in range(HEIGHT):
            res[i][j] = round(scaleFact * (img[i][j] - _min))

    return res


# read image from file to array
images = [np.fromfile(open(dataset + img + '.bin'), dtype=np.uint8).reshape(WIDTH, HEIGHT) for img in images_file_name]

# show the original image,
# real part of the centered DFT,
# the imaginary part of the centered DFT
# the centered DFT log-magnitude spectrum,
# and the phase of centered DFT as 8 bpp images
# with full-scale contrast


def L2Norm(x, y):
    return pow(pow(x, 2.0) + pow(y, 2.0), 0.5)


for i in range(num_of_image):
    X = images[i]  # original image
    show_image(ROWS, COLS, 1, images_file_name[i], X)

    # the centered DFT
    DFT_X = np.fft.fftshift(np.fft.fft2(X))
    XtildeR = DFT_X.real  # real part
    XtildeI = DFT_X.imag  # imaginary part

    # the centered DFT log-magnitude spectrum
    # the phase of centered DFT
    XtildeMag = np.zeros((WIDTH, HEIGHT))
    XtildePhase = np.zeros((WIDTH, HEIGHT))
    for m in range(WIDTH):
        for n in range(HEIGHT):
            XtildeMag[m][n] = np.log2(1.0 + L2Norm(XtildeR[m][n], XtildeI[m][n]))
            XtildePhase[m][n] = math.atan2(XtildeR[m][n], XtildeI[m][n])

    # full scale contrast
    XtildeR = fullScaleContrast(XtildeR)
    XtildeI = fullScaleContrast(XtildeI)
    XtildeMag = fullScaleContrast(XtildeMag)
    XtildePhase = fullScaleContrast(XtildePhase)
    show_image(ROWS, COLS, 3, 'Re[DFT(' + images_file_name[i] + ')]', XtildeR)
    show_image(ROWS, COLS, 4, 'Im[DFT(' + images_file_name[i] + ')]', XtildeI)
    show_image(ROWS, COLS, 5, 'Log-Magnitude Spectrum', XtildeMag)
    show_image(ROWS, COLS, 6, 'arg[DFT(' + images_file_name[i] + ')]', XtildePhase)

    plt.show()
