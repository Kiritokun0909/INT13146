""""
    INT 13146 - Xu Ly Anh
    Homework 4.1
    Ho Duc Hoang - N20DCCN018 - D20CQCHT01-N
"""

import numpy as np
import matplotlib.pyplot as plt

SIZE = 8    # define size image
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
    - I1(m, n) = 0.5 * exp[j.2.pi/8.(u0.m + v0.n)]
    - e^ix = cos x + i.sin x
    - exp[j.2.pi/8.(u0.m + v0.n)] 
    = cos[2.pi/8.(u0.m + v0.n)] + sin[2.pi/8.(u0.m + v0.n)]
            real part                   imaginary part
"""
# initialize image I1
I1 = 0.5 * np.exp(1j * 2 * np.pi / 8 * (u0 * COLS + v0 * ROWS))

# show real and imaginary parts of I1 as grayscale images
# with 8 bits per pixel (bpp)
# and full-scale contrast
plt.subplot(121)
plt.title('Re[I1]')
plt.imshow(fullScaleContrast(np.real(I1), SIZE), cmap='gray')
plt.axis('off')

plt.subplot(122)
plt.title('Im[I1]')
plt.imshow(fullScaleContrast(np.imag(I1), SIZE), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Compute the DFT I1
I1tilde = np.fft.fft2(I1)
I1tilde = np.fft.fftshift(I1tilde)  # center it

I1tildeR = np.round(np.real(I1tilde[:SIZE][:SIZE]) * 10**4) * 10**(-4)
I1tildeI = np.round(np.imag(I1tilde[:SIZE][:SIZE]) * 10**4) * 10**(-4)

print('Re[DFT(I1)]:')
print(I1tildeR)
print('')
print('Im[DFT(I1)]:')
print(I1tildeI)
