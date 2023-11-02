""""
    INT 13146 - Xu Ly Anh
    Homework 4.3
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


# initialize image I3 arrays
I3R = np.zeros((MAX_SIZE, MAX_SIZE))  # real part of image I3
I3I = np.zeros((MAX_SIZE, MAX_SIZE))  # imaginary part of image I3
u0, v0 = 2.0, 2.0

"""
    - I3(m, n) = cos(2.pi.(u0.m + v0.n)/8)
"""
# set the pixel values
for m in range(MAX_SIZE):
    for n in range(MAX_SIZE):
        I3R[m][n] = 0.5 * np.cos(2 * np.pi / 8.0 * (u0 * m + v0 * n))
        I3I[m][n] = 0

# show real and imaginary parts of I3 as grayscale images
# with 8 bits per pixel (bpp)
# and full-scale contrast
plt.subplot(221)
plt.title('Re[I3]')
plt.axis('off')
I3R = fullScaleContrast(I3R)
plt.imshow(I3R, cmap='gray')

# Compute the DFT I3
DFT_I3 = np.fft.fftshift(np.fft.fft2(I3R + I3I, norm='forward'))
DFT_I3R = DFT_I3.real
DFT_I3I = DFT_I3.imag

# for i in range(MAX_SIZE):
#     for j in range(MAX_SIZE):
#         DFT_I3R[i][j] = round(DFT_I3R[i][j])
#         DFT_I3I[i][j] = round(DFT_I3I[i][j])

print('-----------------------------------')
print('Re[DFT(I3)]:')
print(DFT_I3R)
print('-----------------------------------')
print('Im[DFT(I3)]:')
print(DFT_I3I)

plt.show()
