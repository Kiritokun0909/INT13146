""""
    INT 13146 - Xu Ly Anh
    Homework 4.3
    Ho Duc Hoang - N20DCCN018 - D20CQCHT01-N
"""

import numpy as np
import matplotlib.pyplot as plt

SIZE = 8    # define image size
u1, v1 = 1.5, 1.5  # Frequencies
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
    - I5(m, n) = cos(2.pi/8.(u1.m + v1.n))
    just the real part (no need show imaginary part)
"""
# initialize image I5
I5 = np.cos(2 * np.pi / 8 * (u1 * COLS + v1 * ROWS))

# show real and imaginary parts of I5 as grayscale images
# with 8 bits per pixel (bpp)
# and full-scale contrast
plt.subplot(111)
plt.title('I5')
plt.axis('off')
plt.imshow(fullScaleContrast(I5, SIZE), cmap='gray')
plt.show()

# Compute the DFT I5
I5tilde = np.fft.fft2(I5)
I5tilde = np.fft.fftshift(I5tilde)  # center it

I5tildeR = np.round(np.real(I5tilde[:SIZE][:SIZE]) * 10**4) * 10**(-4)
I5tildeI = np.round(np.imag(I5tilde[:SIZE][:SIZE]) * 10**4) * 10**(-4)

print('-----------------------------------')
print('Re[DFT(I5)]:')
print(I5tildeR)
print('-----------------------------------')
print('Im[DFT(I5)]:')
print(I5tildeI)


