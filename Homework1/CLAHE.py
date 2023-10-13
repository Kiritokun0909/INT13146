# INT13146 Image Processing (Xu ly Anh)
# Homework 1.1
# Contrast-limited histogram equalization
# Ho Duc Hoang (N20DCCN018)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# input file
image_file_name = 'moon.jpg'
path = 'dataset/' + image_file_name


# read image
img = cv.imread(path, cv.IMREAD_GRAYSCALE)

# show source image
plt.subplot(121)
plt.title('Source image')
plt.imshow(img, cmap='gray')

# create CLAHE models
limit = 3  # sets threshold for contrast limiting
clahe16 = cv.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))

# apply clahe to image
dst16 = clahe16.apply(img)

# show results
plt.subplot(122)
plt.title(f'16x16 tiles with limit={limit}')
plt.imshow(dst16, cmap='gray')

plt.show()
