""""
    INT 13146 - Xu Ly Anh
    Homework 4.1
    Ho Duc Hoang - N20DCCN018 - D20CQCHT01-N
"""
import cv2
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


# initialize image I1 arrays
I1R = np.zeros((MAX_SIZE, MAX_SIZE))  # real part of image I1
I1I = np.zeros((MAX_SIZE, MAX_SIZE))  # imaginary part of image I1
u0, v0 = 2.0, 2.0

"""
    - I1(m, n) = 0.5 * exp[j.2.pi.(u0.m + v0.n)/8]
    - e^ix = cos x + i.sin x
    - exp[j.2.pi.(u0.m + v0.n)/8] 
    = cos[2.pi.(u0.m + v0.n)/8] + sin[2.pi.(u0.m + v0.n)/8]
            real part                   imaginary part
"""
# set the pixel values
for m in range(MAX_SIZE):
    for n in range(MAX_SIZE):
        I1R[m][n] = 0.5 * np.cos(2 * np.pi / 8.0 * (u0 * m + v0 * n))
        I1I[m][n] = 0.5 * np.sin(2 * np.pi / 8.0 * (u0 * m + v0 * n))

# show real and imaginary parts of I1 as grayscale images
# with 8 bits per pixel (bpp)
# and full-scale contrast
plt.subplot(221)
plt.title('Re[I1]')
plt.axis('off')
I1R = fullScaleContrast(I1R)
plt.imshow(I1R, cmap='gray')

plt.subplot(222)
plt.title('Im[I1]')
plt.axis('off')
I1I = fullScaleContrast(I1I)
plt.imshow(I1I, cmap='gray')

# Compute the DFT I1
DFT_I1 = np.fft.fftshift(np.fft.fft2(I1R + I1I, norm='forward'))
DFT_I1R = DFT_I1.real
DFT_I1I = DFT_I1.imag

# for i in range(MAX_SIZE):
#     for j in range(MAX_SIZE):
#         DFT_I1R[i][j] = round(DFT_I1R[i][j])
#         DFT_I1I[i][j] = round(DFT_I1I[i][j])

print('-----------------------------------')
print('Re[DFT(I1)]:')
print(DFT_I1R)
print('-----------------------------------')
print('Im[DFT(I1)]:')
print(DFT_I1I)

plt.show()
