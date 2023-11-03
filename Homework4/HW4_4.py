""""
    INT 13146 - Xu Ly Anh
    Homework 4.3
    Ho Duc Hoang - N20DCCN018 - D20CQCHT01-N
"""

import numpy as np
import matplotlib.pyplot as plt

SIZE = 8    # define image size
u0, v0 = 2, 2  # Frequencies
ROWS, COLS = np.meshgrid(np.arange(SIZE), np.arange(SIZE))


def fullScaleContrast(__img, size):
    res = np.zeros((size, size))
    _min, _max = np.min(__img), np.max(__img)

    scaleFact = 0
    if (_max - _min) != 0:
        scaleFact = 255.0 / (_max - _min)

    for u in range(size):
        for v in range(size):
            res[u][v] = round(scaleFact * (__img[u][v] - _min))

    return res


"""
    - I4(m, n) = sin(2.pi/8.(u0.m + v0.n))
    just the real part (no need show imaginary part)
"""
# initialize image I4
I4 = np.sin(2 * np.pi / 8 * (u0 * COLS + v0 * ROWS))

# show real and imaginary parts of I4 as grayscale images
# with 8 bits per pixel (bpp)
# and full-scale contrast
plt.subplot(111)
plt.title('I4')
plt.axis('off')
plt.imshow(fullScaleContrast(I4, SIZE), cmap='gray')
plt.show()

# Compute the DFT I4
I4tilde = np.fft.fft2(I4)
I4tilde = np.fft.fftshift(I4tilde)  # center it

I4tildeR = np.round(np.real(I4tilde[:SIZE][:SIZE]) * 10**4) * 10**(-4)
I4tildeI = np.round(np.imag(I4tilde[:SIZE][:SIZE]) * 10**4) * 10**(-4)

print('-----------------------------------')
print('Re[DFT(I4)]:')
print(I4tildeR)
print('-----------------------------------')
print('Im[DFT(I4)]:')
print(I4tildeI)


