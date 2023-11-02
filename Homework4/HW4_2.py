""""
    INT 13146 - Xu Ly Anh
    Homework 4.2
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


# initialize image I2 arrays
I2R = np.zeros((MAX_SIZE, MAX_SIZE))  # real part of image I2
I2I = np.zeros((MAX_SIZE, MAX_SIZE))  # imaginary part of image I2
u0, v0 = 2.0, 2.0

"""
    - I2(m, n) = 0.5 * exp[-j.2pi/8.(u0.m + v0.n)]
    - e^ix = cos x + i.sin x
    - exp[j.2pi/8.(u0.m + v0.n)] 
    = cos[2pi/8.(u0.m + v0.n)] - sin[2pi/8.(u0.m + v0.n)]
            real part                   imaginary part
"""
# set the pixel values
for m in range(MAX_SIZE):
    for n in range(MAX_SIZE):
        I2R[m][n] = 0.5 * np.cos(2 * np.pi / 8.0 * (u0 * m + v0 * n))
        I2I[m][n] = - 0.5 * np.sin(2 * np.pi / 8.0 * (u0 * m + v0 * n))

# show real and imaginary parts of I2 as grayscale images
# with 8 bits per pixel (bpp)
# and full-scale contrast
plt.subplot(221)
plt.title('Re[I2]')
plt.axis('off')
I2R = fullScaleContrast(I2R)
plt.imshow(I2R, cmap='gray')

plt.subplot(222)
plt.title('Im[I2]')
plt.axis('off')
I2I = fullScaleContrast(I2I)
plt.imshow(I2I, cmap='gray')

# Compute the DFT I2
DFT_I2 = np.fft.fftshift(np.fft.fft2(I2R + I2I, norm='forward'))
DFT_I2R = DFT_I2.real
DFT_I2I = DFT_I2.imag

# for i in range(MAX_SIZE):
#     for j in range(MAX_SIZE):
#         DFT_I2R[i][j] = round(DFT_I2R[i][j])
#         DFT_I2I[i][j] = round(DFT_I2I[i][j])

print('-----------------------------------')
print('Re[DFT(I2)]:')
print(DFT_I2R)
print('-----------------------------------')
print('Im[DFT(I2)]:')
print(DFT_I2I)

plt.show()
