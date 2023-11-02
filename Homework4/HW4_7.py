""""
    INT 13146 - Xu Ly Anh
    Homework 4.7
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
dataset = 'dataset/camera.bin'


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


def L2Norm(x, y):
    return pow(pow(x, 2.0) + pow(y, 2.0), 0.5)


I6 = np.fromfile(open(dataset), dtype=np.uint8).reshape((WIDTH, HEIGHT))

DFT_I6 = np.fft.fftshift(np.fft.fft2(I6, norm='forward'))
I6tildeR = DFT_I6.real
I6tildeI = DFT_I6.imag

I6tildeMag = np.zeros((WIDTH, HEIGHT))
I6tildePhase = np.zeros((WIDTH, HEIGHT))

for m in range(WIDTH):
    for n in range(HEIGHT):
        I6tildeMag[m][n] = np.log2(1.0 + L2Norm(I6tildeR[m][n], I6tildeI[m][n]))
        I6tildePhase[m][n] = math.atan2(I6tildeR[m][n], I6tildeI[m][n])

J1tildeMag = np.zeros((WIDTH, HEIGHT))
J1tildePhase = np.zeros((WIDTH, HEIGHT))

J1tildeR = np.zeros((WIDTH, HEIGHT))
J1tildeI = np.zeros((WIDTH, HEIGHT))

J2tildeMag = np.zeros((WIDTH, HEIGHT))
J2tildePhase = np.zeros((WIDTH, HEIGHT))

J2tildeR = np.zeros((WIDTH, HEIGHT))
J2tildeI = np.zeros((WIDTH, HEIGHT))
for m in range(WIDTH):
    for n in range(HEIGHT):
        J1tildeMag[m][n] = I6tildeMag[m][n]
        J1tildePhase[m][n] = 0.0
        J1tildeR[m][n] = J1tildeMag[m][n] * np.cos(J1tildePhase[m][n])
        J1tildeI[m][n] = J1tildeMag[m][n] * np.sin(J1tildePhase[m][n])

        J2tildeMag[m][n] = 1.0
        J2tildePhase[m][n] = I6tildePhase[m][n]
        J2tildeR[m][n] = J2tildeMag[m][n] * np.cos(J2tildePhase[m][n])
        J2tildeI[m][n] = J2tildeMag[m][n] * np.sin(J2tildePhase[m][n])


J1 = np.fft.ifft2(J1tildeR + J1tildeI, norm='backward')
J2 = np.fft.ifft2(J2tildeR + J2tildeI, norm='backward')

for i in range(WIDTH):
    for j in range(HEIGHT):
        J1[i][j] = np.log(J1[i][j])


J1, J2 = J1.real, J2.real
J1 = fullScaleContrast(J1)
J2 = fullScaleContrast(J2)


def show_image(rows, cols, pos, title, img):
    plt.subplot(rows, cols, pos)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img, cmap='gray')


show_image(2, 2, 1, 'J1prime', J1)
show_image(2, 2, 2, 'J2', J2)

plt.show()
