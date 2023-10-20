# INT13146 Image Processing (Xu ly Anh)
# Homework 3.1
# Ho Duc Hoang (N20DCCN018)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imutils


def show_image(row, col, pos, image, title):
    plt.subplot(row, col, pos)
    plt.imshow(image, cmap='gray')
    plt.title(title)


# Construct an array from data in binary file
w, h = 256, 256
arr = np.fromfile(open('dataset/Mammogram.bin'), dtype=np.uint8).reshape(w, h)
image = Image.fromarray(arr)
show_image(2, 2, 1, image, 'Source image')

# (a): Convert gray scale image into a binary image by simple thresholding
img = arr.astype(float)
_, thresh = cv2.threshold(img, 95, 255, cv2.THRESH_BINARY)

plt.subplot(2, 2, 2)
plt.imshow(thresh, 'gray', vmin=0, vmax=255)
plt.title('Binary image')

# (b): implement the Approximate Contour Image Generation algorithm
contours = thresh.copy()
for i in range(1, len(thresh)):
    for j in range(1, len(thresh)):
        if contours[i][j - 1] != contours[i][j]:
            # draw contours with thickness = 2
            contours[i][j + 1] = contours[i][j]
            for z in range(j + 2, len(thresh)):
                contours[i][z] = contours[i][j - 1]
            break

# show result
plt.subplot(2, 2, 3)
plt.imshow(contours, 'gray')
plt.title('Contours binary image')

plt.show()
