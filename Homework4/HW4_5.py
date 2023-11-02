""""
    INT 13146 - Xu Ly Anh
    Homework 4.5
    Ho Duc Hoang - N20DCCN018 - D20CQCHT01-N
"""

import numpy as np
import matplotlib.pyplot as plt

# define constant size of image
MAX_SIZE = 8

def fullScaleContrast(img):
    res = np.zeros((MAX_SIZE, MAX_SIZE))
    _min = np.min(img)
    _max = np.max(img)

    scaleFact = 0
    if (_max - _min) != 0:
        scaleFact = 255.0 / (_max - _min)

    for i in range(MAX_SIZE):
        for j in range(MAX_SIZE):
            res[i][j] = round(scaleFact * (img[i][j] - _min))

    return res


# initialize image I5 arrays
I5 = np.zeros((MAX_SIZE, MAX_SIZE))  # image I5
u1, v1 = 1.5, 1.5

"""
    - I5(m, n) = cos(2.pi.(u1.m + v1.n)/8)
"""
# set the pixel values
for m in range(MAX_SIZE):
    for n in range(MAX_SIZE):
        I5[m][n] = 0.5 * np.cos(2 * np.pi / 8.0 * (u1 * m + v1 * n))

# show real and imaginary parts of I5 as grayscale images
# with 8 bits per pixel (bpp)
# and full-scale contrast
plt.subplot(111)
plt.title('I5')
plt.axis('off')
I5 = fullScaleContrast(I5)
plt.imshow(I5, cmap='gray')

# Compute the DFT I5
DFT_I5 = np.fft.fftshift(np.fft.fft2(I5, norm='forward'))
DFT_I5R = DFT_I5.real
DFT_I5I = DFT_I5.imag

# for i in range(MAX_SIZE):
#     for j in range(MAX_SIZE):
#         DFT_I5R[i][j] = round(DFT_I5R[i][j])
#         DFT_I5I[i][j] = round(DFT_I5I[i][j])

print('-----------------------------------')
print('Re[DFT(I5)]:')
print(DFT_I5R)
print('-----------------------------------')
print('Im[DFT(I5)]:')
print(DFT_I5I)


plt.show()
