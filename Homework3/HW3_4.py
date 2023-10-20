# INT13146 Image Processing (Xu ly Anh)
# Homework 3.2
# Ho Duc Hoang (N20DCCN018)

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# input file path
file_path = 'dataset/johnny.bin'


def show_image(row, col, pos, image, title):
    plt.subplot(row, col, pos)
    plt.imshow(image, cmap='gray')
    plt.title(title)


# Construct an array from data in binary file
w, h = 256, 256
arr = np.fromfile(open(file_path), dtype=np.uint8).reshape(w, h)

img = Image.fromarray(arr)
show_image(2, 2, 1, img, 'johnny.bin')

histr = cv2.calcHist([arr], [0], None, [256], [0, 256])
plt.subplot(2, 2, 2)
plt.plot(histr)
plt.title('Histogram of Original Johnny Image')

dst = cv2.equalizeHist(arr)
show_image(2, 2, 3, dst, 'Histogram Equalized Image')

# plot the histogram
histr2 = cv2.calcHist([dst], [0], None, [256], [0, 256])
plt.subplot(2, 2, 4)
plt.plot(histr2)
plt.title('Histogram of Equalized Johnny Image')

plt.show()
