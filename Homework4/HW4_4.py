""""
    INT 13146 - Xu Ly Anh
    Homework 4.4
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


# initialize image I4 arrays
I4R = np.zeros((MAX_SIZE, MAX_SIZE))  # real part of image I4
I4I = np.zeros((MAX_SIZE, MAX_SIZE))  # imaginary part of image I4
u0, v0 = 2.0, 2.0

"""
    - I4(m, n) = sin(2.pi.(u0.m + v0.n)/8)
"""
# set the pixel values
for m in range(MAX_SIZE):
    for n in range(MAX_SIZE):
        I4R[m][n] = 0.5 * np.sin(2 * np.pi / 8.0 * (u0 * m + v0 * n))
        I4I[m][n] = 0

# show real and imaginary parts of I4 as grayscale images
# with 8 bits per pixel (bpp)
# and full-scale contrast
plt.subplot(221)
plt.title('Re[I4]')
plt.axis('off')
I4R = fullScaleContrast(I4R)
plt.imshow(I4R, cmap='gray')

# Compute the DFT I4
DFT_I4 = np.fft.fftshift(np.fft.fft2(I4R + I4I, norm='forward'))
DFT_I4R = DFT_I4.real
DFT_I4I = DFT_I4.imag

for i in range(MAX_SIZE):
    for j in range(MAX_SIZE):
        DFT_I4R[i][j] = round(DFT_I4R[i][j])
        DFT_I4I[i][j] = round(DFT_I4I[i][j])

print('-----------------------------------')
print('Re[DFT(I4)]:')
print(DFT_I4R)
print('-----------------------------------')
print('Im[DFT(I4)]:')
print(DFT_I4I)

plt.show()
