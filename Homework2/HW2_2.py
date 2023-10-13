# INT13146 Image Processing (Xu ly Anh)
# Homework 2.2
# Ho Duc Hoang (N20DCCN018)

import cv2 as cv

import matplotlib.pyplot as plt
from PIL import Image

# (a) help imread and imwrite
print(help(cv.imread))
print(help(cv.imwrite))


# (b) obtain file path
path = 'dataset/lenagray.jpg'


# (c) read image
J1 = cv.imread(path, cv.IMREAD_GRAYSCALE)


# (d) display image and write it out as a JPEG file
# make photographic negative of J1
J2 = 255 - J1

# write J2
cv.imwrite('HW22.jpeg', J2)

# display image J2
plt.subplot(111)
plt.title('J2 image')
plt.imshow(J2, cmap='gray')

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
