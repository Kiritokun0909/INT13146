# INT13146 Image Processing (Xu ly Anh)
# Homework 3.3
# Ho Duc Hoang (N20DCCN018)

import numpy as np
import matplotlib.pyplot as plt
import cv2

# input file path
file_path = 'dataset/actontBin.bin'


def show_image(row, col, pos, image, title):
    plt.subplot(row, col, pos)
    plt.imshow(image, cmap='gray')
    plt.title(title)


# Construct an array from data in binary file
w, h = 256, 256
arr = np.fromfile(open(file_path), dtype=np.uint8).reshape(w, h)

image = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
show_image(2, 2, 1, image, 'actontBin.bin')

# Build an image of the letter T
imgT = np.zeros((30, 22))
imgT[2:7, 2:20] = 255
imgT[7:27, 9:13] = 255
show_image(2, 2, 2, imgT, 'T letter')

for i in range(h):
    for j in range(w):
        

# Binary Template Matching algorithm



plt.show()
