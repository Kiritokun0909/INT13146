# INT13146 Image Processing (Xu ly Anh)
# Homework 2.1
# Ho Duc Hoang (N20DCCN018)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# input file path
path = 'dataset/'
image_file_name = ['lena.bin', 'peppers.bin']


def show_image(row, col, pos, image, title):
    plt.subplot(row, col, pos)
    plt.imshow(image, cmap='gray')
    plt.title(title)


# (a) read and display the images

# define width and height image
w, h = 256, 256

# Construct an array from data in binary file
arr = [np.fromfile(open(path + filename), dtype=np.uint8).reshape(w, h) for filename in image_file_name]

images = [Image.fromarray(i) for i in arr]

show_image(2, 2, 1, images[0], f'{image_file_name[0]}')
show_image(2, 2, 2, images[1], f'{image_file_name[1]}')


# (b) left half lena, right half peppers
JA = np.zeros((w, h))
for i in range(256):
    a = list(arr[0][i][:129]) + list(arr[1][i][129:])
    JA[i] = a

J = Image.fromarray(JA)
show_image(2, 2, 3, J, 'J image')


# (c) swap left and right half of K image
KA = np.zeros((w, h))
for i in range(h):
    arr = list(JA[i][129:]) + list(JA[i][:129])
    KA[i] = arr

K = Image.fromarray(KA)
show_image(2, 2, 4, K, 'K image')


# (d) showing
plt.show()
