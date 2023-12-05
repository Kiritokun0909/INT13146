"""
    INT13146 - Image Processing (Xu ly anh)
    Homework 5.1
    Student: Ho Duc Hoang (N20DCCN018)
    Class: D20CQCNHT01-N
"""

import numpy as np
import matplotlib.pyplot as plt

# read image salesman from bin file with size 256x256
salesman = np.fromfile(open('dataset/salesman.bin'), dtype=np.uint8).reshape(256, 256)

# create filter 7x7 square where each pixel equal to 1/49
kernel7 = np.ones((7, 7)) / 49


# print(kernel7)


# defined function to use in this homework
def fullScaleContrast(x):
    xMax = np.max(x)
    xMin = np.min(x)
    if xMax - xMin == 0:
        return np.zeros(x.shape)
    Scale_factor = 255.0 / (xMax - xMin)
    return np.round((x - xMin) * Scale_factor)


def show_image(row, col, pos, image, title):
    plt.subplot(row, col, pos)
    plt.imshow(fullScaleContrast(image), cmap='gray')
    plt.title(title, loc='center', wrap=True)
    plt.axis('off')


# --------------------------------------------------------------
# (a):  Implement the 7×7 linear average filter using image domain convolution. Set each
# pixel in the output image equal to the average of the pixels in a 7×7 neighborhood
# about the corresponding pixel in the input image.
def convolve(_image, _kernel, mode='full'):
    M, N = _image.shape[0], _image.shape[1]  # image size M x N
    m, n = _kernel.shape[0], _kernel.shape[1]  # kernel size m x n

    # Handle edge effects by zero padding
    # padded_image will use a value of zero for pixels
    # where the window “hangs over” the edges of the input image
    if mode == 'full':
        res = np.zeros((M + m - 1, N + n - 1))
        padded_image = np.zeros((M + m // 2 * 4, N + n // 2 * 4))
        padded_image[m // 2 * 2: m // 2 * 2 + M, n // 2 * 2: n // 2 * 2 + N] = _image
    elif mode == 'same':  #
        res = np.zeros((M, N))
        padded_image = np.zeros((M + m // 2 * 2, N + n // 2 * 2))
        padded_image[m // 2: m // 2 + M, n // 2: n // 2 + N] = _image
    elif mode == 'valid':
        res = np.zeros((max(M - m + 1, 0), max(N - n + 1, 0)))
        padded_image = _image
    else:
        return None

    w, h = res.shape[0], res.shape[1]
    for i in range(w):
        for j in range(h):
            window = padded_image[i:i + m, j:j + n]
            res[i][j] = np.sum(window * _kernel)

    return res


# Image convolution using mode 'same' to return image
# with same size as input image
M1_filtered_image = convolve(salesman, kernel7, 'same')

# show input & output images
show_image(1, 2, 1, salesman, 'Original image')
show_image(1, 2, 2, M1_filtered_image, 'Filtered image')

M1_filtered_image = fullScaleContrast(M1_filtered_image)
plt.show()

# --------------------------------------------------------------
# (b): Implement the same filter by pointwise multiplication of DFT’s using the method
# of Example 3 on pages 5.61-5.62 of the course notes

# define impulse response image H with size 128x128
# with all pixel equal to zero
# except for a 7x7 square of pixels in the center
# that will be equal to 1/49
# the center of square is at pixel (65, 65)
H = np.zeros((128, 128))
H[62:69, 62:69] = 1 / 49

# original input images
show_image(1, 2, 1, salesman, 'Original image')

# zero padded original image
padded_image = np.zeros((384, 384))
padded_image[0:256, 0:256] = salesman
show_image(1, 2, 2, padded_image, 'Zero padded image')

plt.show()

# zero padded impulse response image
padded_impulse_image = np.zeros((384, 384))
padded_impulse_image[0:128, 0:128] = H
show_image(1, 2, 1, padded_impulse_image, 'Zero padded Impulse Response')

# Compute the DFT of the zero padded input image
DFT_padded_image = np.fft.fft2(padded_image)

# the centered DFT log-magnitude spectrum
padded_image_tildeMag = np.log(np.abs(np.fft.fftshift(DFT_padded_image)) + 1)
show_image(1, 2, 2, padded_image_tildeMag, 'Log-Magnitude Spectrum of zero padded image')

plt.show()

# Compute the DFT of the zero padded impulse response image H
DFT_padded_impulse_image = np.fft.fft2(padded_impulse_image)

# the centered DFT log-magnitude spectrum
padded_impulse_image_tildeMag = np.log(np.abs(np.fft.fftshift(DFT_padded_impulse_image)) + 1)
show_image(1, 2, 1, padded_impulse_image_tildeMag, 'Log-Magnitude Spectrum of zero padded H image')

# Compute the centered DFT of the zero padded output image (filtered image from (a))
DFT_padded_filtered_image = DFT_padded_image * DFT_padded_impulse_image

# the centered DFT log-magnitude spectrum
padded_impulse_image_tildeMag = np.log(np.abs(np.fft.fftshift(DFT_padded_filtered_image)) + 1)
show_image(1, 2, 2, padded_impulse_image_tildeMag, 'Log-Magnitude Spectrum of zero padded output image')

plt.show()

# show zero padded and 256x256 output image
padded_filtered_image = np.real(np.fft.ifft2(DFT_padded_filtered_image))
show_image(1, 2, 1, padded_filtered_image, 'Zero padded output image')

M2_filtered_image = padded_filtered_image[65:321, 65:321]
show_image(1, 2, 2, M2_filtered_image, 'Final 256x256 output image')

M2_filtered_image = fullScaleContrast(M2_filtered_image)
plt.show()

# Verify that the output image (obtained after performing a full-scale contrast
# stretch and rounding each pixel to the nearest integer) is the same as the one
# in part (a)
print(f'(b): max difference from part (a): {np.max(np.abs(M2_filtered_image - M1_filtered_image))}')

# --------------------------------------------------------------
# (c): Implement the same filter again using the zero-phase impulse response and DFT
# method of Example 5 on page 5.76 of the course notes

# define impulse response image H with size 256x256
# with all pixel equal to zero
# except for a 7x7 square of pixels in the center
# that will be equal to 1/49
# the center of square is at pixel (129, 129)
H = np.zeros((256, 256))
H[125:132, 125:132] = 1 / 49

# original input images
show_image(1, 2, 1, salesman, 'Original image')

# The 256 × 256 zero-phase impulse response image
H2 = np.fft.fftshift(H)
show_image(1, 2, 2, H2, 'zero-phase impulse response image')

plt.show()

# The 512 × 512 zero padded zero-phase impulse response image
H2ZP = np.zeros((512, 512))
H2ZP[0:128, 0:128] = H2[0:128, 0:128]
H2ZP[0:128, 384:512] = H2[0:128, 128:256]
H2ZP[384:512, 0:128] = H2[128:256, 0:128]
H2ZP[384:512, 384:512] = H2[128:256, 128:256]
show_image(1, 2, 1, H2ZP, 'zero padded zero-phase impulse response image')

#  The final 256 × 256 output image
padded_image = np.zeros((512, 512))
padded_image[0:256, 0:256] = salesman

padded_filtered_image = np.fft.ifft2(np.fft.fft2(padded_image) * np.fft.fft2(H2ZP))
M3_filtered_image = np.real(padded_filtered_image)
M3_filtered_image = M3_filtered_image[0:256, 0:256]
show_image(1, 2, 2, M3_filtered_image, 'Final 256x256 output image')
M3_filtered_image = fullScaleContrast(M3_filtered_image)
plt.show()

# Verify that the output image (obtained after performing a full-scale contrast
# stretch and rounding each pixel to the nearest integer) is the same as the one
# in part (a)
print(f'(c): max difference from part (a): {np.max(np.abs(M3_filtered_image - M1_filtered_image))}')

