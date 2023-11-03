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
SIZE = 256


def show_image(rows, cols, pos, title, img):
    plt.subplot(rows, cols, pos)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img, cmap='gray')


def fullScaleContrast(x):
    xMax = np.max(x)
    xMin = np.min(x)
    if xMax - xMin == 0:
        return np.zeros(x.shape)
    Scale_factor = 255.0 / (xMax - xMin)
    return np.round((x - xMin) * Scale_factor)


# read image
I6 = np.fromfile(open('dataset/camera.bin'), dtype=np.uint8).reshape((SIZE, SIZE))

# Compute the DFT I6
DFT_I6 = np.fft.fft2(I6)

J1tilde = np.abs(DFT_I6)
J1 = np.real(np.fft.ifft2(J1tilde))
J1prime = fullScaleContrast(np.log(1 + J1))
show_image(1, 2, 1, 'J1prime', J1prime)

J2tilde = np.exp(1j * np.angle(DFT_I6))
J2 = fullScaleContrast(np.real(np.fft.ifft2(J2tilde)))
show_image(1, 2, 2, 'J2', J2)

plt.show()
