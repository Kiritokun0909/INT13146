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
    - I3(m, n) = cos(2.pi/8.(u0.m + v0.n))
    just the real part (no need show imaginary part)
"""
# initialize image I3
I3 = np.cos(2 * np.pi / 8 * (u0 * COLS + v0 * ROWS))

# show real and imaginary parts of I3 as grayscale images
# with 8 bits per pixel (bpp)
# and full-scale contrast
plt.subplot(111)
plt.title('I3')
plt.axis('off')
plt.imshow(fullScaleContrast(I3, SIZE), cmap='gray')
plt.show()

# Compute the DFT I3
I3tilde = np.fft.fft2(I3)
I3tilde = np.fft.fftshift(I3tilde)  # center it

I3tildeR = np.round(np.real(I3tilde[:SIZE][:SIZE]) * 10**4) * 10**(-4)
I3tildeI = np.round(np.imag(I3tilde[:SIZE][:SIZE]) * 10**4) * 10**(-4)

print('-----------------------------------')
print('Re[DFT(I3)]:')
print(I3tildeR)
print('-----------------------------------')
print('Im[DFT(I3)]:')
print(I3tildeI)


