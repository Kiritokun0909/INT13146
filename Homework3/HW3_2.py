# INT13146 Image Processing (Xu ly Anh)
# Homework 3.2
# Ho Duc Hoang (N20DCCN018)

import numpy as np
import matplotlib.pyplot as plt
import cv2

# input file path
file_path = 'dataset/lady.bin'


def show_image(row, col, pos, image, title):
    plt.subplot(row, col, pos)
    plt.imshow(image, cmap='gray')
    plt.title(title)


# Construct an array from data in binary file
w, h = 256, 256
arr = np.fromfile(open(file_path), dtype=np.uint8).reshape(w, h)

image = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
show_image(2, 2, 1, image, 'lady.bin')

histr = cv2.calcHist(arr, [0], None, [256], [0, 256])
plt.subplot(2, 2, 2)
plt.plot(histr)
plt.title('Histogram of Original Lady')

# perform a full-scale contrast stretch on image
dst = cv2.normalize(arr, None, 0, 255, norm_type=cv2.NORM_MINMAX)
show_image(2, 2, 3, dst, 'full-scale stretch')

# plot the histogram
histr2 = cv2.calcHist(dst, [0], None, [256], [0, 256])
plt.subplot(2, 2, 4)
plt.plot(histr2)
plt.title('Histogram of Lady After Full-scale stretch')

plt.show()
