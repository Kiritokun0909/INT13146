# INT13146 Image Processing (Xu ly Anh)
# Homework 1.2
# Adaptive histogram equalization
# Ho Duc Hoang (N20DCCN018)

import cv2 as cv
import matplotlib.pyplot as plt

# input file path
path = 'dataset/'
image_file_name = [
    'dental.jpg'
    # ,'parrot.jpg'
    # ,'skull.jpg'
]


def show_image(row, col, pos, image, title):
    plt.subplot(row, col, pos)
    plt.imshow(image, cmap='gray')
    plt.title(title)


def show_plot(row, col, pos, plot, title):
    plt.subplot(row, col, pos)
    plt.plot(plot)
    plt.title(title)


# read image
src = [cv.imread(path + filename, cv.IMREAD_GRAYSCALE) for filename in image_file_name]

# calculate histogram of source image
histr = [cv.calcHist(src_img, [0], None, [256], [0, 256]) for src_img in src]

# perform adaptive histogram equalization
clahe16 = cv.createCLAHE(clipLimit=0)  # clipLimit = 0 to ensure no contrast-limited
dst = [clahe16.apply(src_img) for src_img in src]

# calculate histogram of destination image
# find frequency of pixels in range 0-255
histr2 = [cv.calcHist(dst_img, [0], None, [256], [0, 256]) for dst_img in dst]

src_size = len(image_file_name)
num_row = src_size * 2
num_col = 2
j = 0
for i in range(src_size):
    show_image(num_row, num_col, j + 1, src[i], image_file_name[i])
    show_plot(num_row, num_col, j + 2, histr[i], image_file_name[i] + ' gray level')

    show_image(num_row, num_col, j + 3, dst[i], image_file_name[i] + ' after AHE')
    show_plot(num_row, num_col, j + 4, histr2[i], image_file_name[i] + ' after AHE gray level')

    j += 4

plt.show()
