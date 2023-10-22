# INT13146 Image Processing (Xu ly Anh)
# Homework 3.3
# Ho Duc Hoang (N20DCCN018)

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load image
w, h = 256, 256
arr = np.fromfile(open('dataset/actontBin.bin'), dtype=np.uint8).reshape(w, h) / 255.0

# show original image
plt.subplot(2, 2, 1)
plt.imshow(arr, cmap='gray')
plt.title('actontBin.bin')

# Build template of the letter T
TemplRows, TemplCols = 47, 15
template = np.zeros((TemplRows, TemplCols)).astype(np.uint8)
template[10:16, :] = 255
template[16:37, 6:10] = 255

# show template
plt.subplot(2, 2, 2)
plt.imshow(template, cmap='gray')
plt.title('T letter')

# Binary Template Matching algorithm
J1 = np.zeros((w, h))
for row in range(h - TemplRows):
    for col in range(w - TemplCols):
        cmp_area = arr[row: row + TemplRows, col: col + TemplCols]

        # xor: 2 pixels have the same value is True, else is False
        matching_template = np.logical_not(np.logical_xor(cmp_area, template))

        # sum of number of True
        J1[row, col] = np.sum(matching_template)

# show result after algorithm
plt.subplot(2, 2, 3)
plt.imshow(J1, cmap='gray')
plt.title('J1 = M2(i, j)')

# show max True value can get in Template Matching
print(np.max(J1))

# thresholding J1 to J2
_, J2 = cv2.threshold(src=J1, thresh=685, maxval=705, type=cv2.THRESH_BINARY)

plt.subplot(2, 2, 4)
plt.imshow(J2, cmap='gray')
plt.title('J2')

plt.show()
