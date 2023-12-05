"""
    INT13146 - Image Processing (Xu ly anh)
    Homework 5.2
    Student: Ho Duc Hoang (N20DCCN018)
    Class: D20CQCNHT01-N
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def fullScaleContrast(x):
    xMax = np.max(x)
    xMin = np.min(x)
    if xMax - xMin == 0:
        return np.zeros(x.shape)
    Scale_factor = 255.0 / (xMax - xMin)
    return np.round((x - xMin) * Scale_factor)


def show_image(row, col, pos, image, title):
    plt.subplot(row, col, pos)
    plt.imshow(image, cmap='gray')
    plt.title(title, loc='center', wrap=True)
    plt.axis('off')


def MSE(x1, x2):
    return np.mean((x1 - x2)**2)


# (a): Read and display all three images
# read images
girl2 = np.fromfile(open('dataset/girl2.bin'), dtype=np.uint8).reshape(256, 256)
girl2Noise32Hi = np.fromfile(open('dataset/girl2Noise32Hi.bin'), dtype=np.uint8).reshape(256, 256)
girl2Noise32 = np.fromfile(open('dataset/girl2Noise32.bin'), dtype=np.uint8).reshape(256, 256)

# show image
show_image(1, 3, 1, girl2, 'Original Tiffany image')
show_image(1, 3, 2, girl2Noise32Hi, 'girl2Noise32Hi.bin')
show_image(1, 3, 3, girl2Noise32, 'girl2Noise32.bin')

# Compute the MSE of girl2Noise32Hi and
# girl2Noise32 relative to girl2 using the formula given on page 5.114 of the
# course notes.
MSE_girl2Noise32Hi = MSE(girl2, girl2Noise32Hi)
MSE_girl2Noise32 = MSE(girl2, girl2Noise32)

print('(a): ')
print(f'MSE girl2Noise32Hi.bin: {MSE_girl2Noise32Hi}')
print(f'MSE girl2Noise32.bin: {MSE_girl2Noise32}')
print('-----------------------------------------------------')
plt.show()

# --------------------------------------------------------------
# (b): Apply an isotropic ideal low-pass filter as on page 5.111 of the course notes with
# Cutoff = 64 cpi to all three images
U_cutoff = 64
U, V = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
HLtildeCenter = np.sqrt(U**2 + V**2) <= U_cutoff
HLtilde = np.fft.fftshift(HLtildeCenter)

# DFT array of each image
filtered_girl2 = np.real(np.fft.ifft2(np.fft.fft2(girl2) * HLtilde))
filtered_girl2Noise32Hi = np.real(np.fft.ifft2(np.fft.fft2(girl2Noise32Hi) * HLtilde))
filtered_girl2Noise32 = np.real(np.fft.ifft2(np.fft.fft2(girl2Noise32) * HLtilde))

show_image(1, 3, 1, fullScaleContrast(filtered_girl2), 'filtered_girl2')
show_image(1, 3, 2, fullScaleContrast(filtered_girl2Noise32Hi), 'filtered_girl2Noise32Hi')
show_image(1, 3, 3, fullScaleContrast(filtered_girl2Noise32), 'filtered_girl2Noise32')

# compute the MSE (relative to the original girl2 image)
MSE_filtered_girl2 = MSE(filtered_girl2, girl2)

MSE_filtered_girl2Noise32Hi = MSE(filtered_girl2Noise32Hi, girl2)
ISNR_filtered_girl2Noise32Hi = 10 * np.log10(MSE_girl2Noise32Hi/MSE_filtered_girl2Noise32Hi)

MSE_filtered_girl2Noise32 = MSE(filtered_girl2Noise32, girl2)
ISNR_filtered_girl2Noise32 = 10 * np.log10(MSE_girl2Noise32/MSE_filtered_girl2Noise32)

print('(b): ')
print(f'MSE filtered_girl2: {MSE_filtered_girl2}')
print(f'MSE filtered_girl2Noise32Hi: {MSE_filtered_girl2Noise32Hi}')
print(f'ISNR filtered_girl2Noise32Hi: {ISNR_filtered_girl2Noise32Hi}')
print(f'MSE filtered_girl2Noise32: {MSE_filtered_girl2Noise32}')
print(f'ISNR filtered_girl2Noise32: {ISNR_filtered_girl2Noise32}')
print('-----------------------------------------------------')
plt.show()

# --------------------------------------------------------------
# (c): Apply the Gaussian low-pass filter with Ucutoff = 64
U_cutoff_H = 64
SigmaH = 0.19 * 256 / U_cutoff_H
U, V = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
HtildeCenter = np.exp((-2*(math.pi**2)*(SigmaH**2))/(256**2)*(U**2 + V**2))
Htilde = np.fft.fftshift(HtildeCenter)

H = np.real(np.fft.ifft2(Htilde))
H2 = np.fft.fftshift(H)

ZPH2 = np.zeros((512, 512))
ZPH2[0:256, 0:256] = H2

# DFT ZPH2
DFT_ZPH2 = np.fft.fft2(ZPH2)

# three zero padded images and compute DFT
ZP_girl2 = np.zeros((512, 512))
ZP_girl2[0:256, 0:256] = girl2

ZP_girl2Noise32Hi = np.zeros((512, 512))
ZP_girl2Noise32Hi[0:256, 0:256] = girl2Noise32Hi

ZP_girl2Noise32 = np.zeros((512, 512))
ZP_girl2Noise32[0:256, 0:256] = girl2Noise32

filtered_girl2 = np.real(np.fft.ifft2(np.fft.fft2(ZP_girl2) * DFT_ZPH2))
filtered_girl2 = filtered_girl2[128:384, 128:384]

filtered_girl2Noise32Hi = np.real(np.fft.ifft2(np.fft.fft2(ZP_girl2Noise32Hi) * DFT_ZPH2))
filtered_girl2Noise32Hi = filtered_girl2Noise32Hi[128:384, 128:384]

filtered_girl2Noise32 = np.real(np.fft.ifft2(np.fft.fft2(ZP_girl2Noise32) * DFT_ZPH2))
filtered_girl2Noise32 = filtered_girl2Noise32[128:384, 128:384]

show_image(1, 3, 1, fullScaleContrast(filtered_girl2), 'filtered_girl2')
show_image(1, 3, 2, fullScaleContrast(filtered_girl2Noise32Hi), 'filtered_girl2Noise32Hi')
show_image(1, 3, 3, fullScaleContrast(filtered_girl2Noise32), 'filtered_girl2Noise32')

# compute the MSE (relative to the original girl2 image)
MSE_filtered_girl2 = MSE(filtered_girl2, girl2)

MSE_filtered_girl2Noise32Hi = MSE(filtered_girl2Noise32Hi, girl2)
ISNR_filtered_girl2Noise32Hi = 10 * np.log10(MSE_girl2Noise32Hi/MSE_filtered_girl2Noise32Hi)

MSE_filtered_girl2Noise32 = MSE(filtered_girl2Noise32, girl2)
ISNR_filtered_girl2Noise32 = 10 * np.log10(MSE_girl2Noise32/MSE_filtered_girl2Noise32)

print('(c): ')
print(f'MSE filtered_girl2: {MSE_filtered_girl2}')
print(f'MSE filtered_girl2Noise32Hi: {MSE_filtered_girl2Noise32Hi}')
print(f'ISNR filtered_girl2Noise32Hi: {ISNR_filtered_girl2Noise32Hi}')
print(f'MSE filtered_girl2Noise32: {MSE_filtered_girl2Noise32}')
print(f'ISNR filtered_girl2Noise32: {ISNR_filtered_girl2Noise32}')
print('-----------------------------------------------------')
plt.show()


# --------------------------------------------------------------
# (d):  apply the Gaussian low-pass filter with Ucutoff = 77.5
U_cutoff_H = 77.5
SigmaH = 0.19 * 256 / U_cutoff_H
U, V = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
HtildeCenter = np.exp((-2*(math.pi**2)*(SigmaH**2))/(256**2)*(U**2 + V**2))
Htilde = np.fft.fftshift(HtildeCenter)

H = np.real(np.fft.ifft2(Htilde))
H2 = np.fft.fftshift(H)

ZPH2 = np.zeros((512, 512))
ZPH2[0:256, 0:256] = H2

# DFT ZPH2
DFT_ZPH2 = np.fft.fft2(ZPH2)

# three zero padded images and compute DFT
ZP_girl2 = np.zeros((512, 512))
ZP_girl2[0:256, 0:256] = girl2

ZP_girl2Noise32Hi = np.zeros((512, 512))
ZP_girl2Noise32Hi[0:256, 0:256] = girl2Noise32Hi

ZP_girl2Noise32 = np.zeros((512, 512))
ZP_girl2Noise32[0:256, 0:256] = girl2Noise32

filtered_girl2 = np.real(np.fft.ifft2(np.fft.fft2(ZP_girl2) * DFT_ZPH2))
filtered_girl2 = filtered_girl2[128:384, 128:384]

filtered_girl2Noise32Hi = np.real(np.fft.ifft2(np.fft.fft2(ZP_girl2Noise32Hi) * DFT_ZPH2))
filtered_girl2Noise32Hi = filtered_girl2Noise32Hi[128:384, 128:384]

filtered_girl2Noise32 = np.real(np.fft.ifft2(np.fft.fft2(ZP_girl2Noise32) * DFT_ZPH2))
filtered_girl2Noise32 = filtered_girl2Noise32[128:384, 128:384]

show_image(1, 3, 1, fullScaleContrast(filtered_girl2), 'filtered_girl2')
show_image(1, 3, 2, fullScaleContrast(filtered_girl2Noise32Hi), 'filtered_girl2Noise32Hi')
show_image(1, 3, 3, fullScaleContrast(filtered_girl2Noise32), 'filtered_girl2Noise32')

# compute the MSE (relative to the original girl2 image)
MSE_filtered_girl2 = MSE(filtered_girl2, girl2)

MSE_filtered_girl2Noise32Hi = MSE(filtered_girl2Noise32Hi, girl2)
ISNR_filtered_girl2Noise32Hi = 10 * np.log10(MSE_girl2Noise32Hi/MSE_filtered_girl2Noise32Hi)

MSE_filtered_girl2Noise32 = MSE(filtered_girl2Noise32, girl2)
ISNR_filtered_girl2Noise32 = 10 * np.log10(MSE_girl2Noise32/MSE_filtered_girl2Noise32)

print('(d): ')
print(f'MSE filtered_girl2: {MSE_filtered_girl2}')
print(f'MSE filtered_girl2Noise32Hi: {MSE_filtered_girl2Noise32Hi}')
print(f'ISNR filtered_girl2Noise32Hi: {ISNR_filtered_girl2Noise32Hi}')
print(f'MSE filtered_girl2Noise32: {MSE_filtered_girl2Noise32}')
print(f'ISNR filtered_girl2Noise32: {ISNR_filtered_girl2Noise32}')
print('-----------------------------------------------------')
plt.show()
