# INT13146 Image Processing (Xu ly Anh)
# Homework 2.3
# Ho Duc Hoang (N20DCCN018)
import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# (b) read and display image
path = 'dataset/lena512color.jpg'
imgJ1 = cv.imread(path)
J1 = cv2.cvtColor(imgJ1, cv2.COLOR_BGR2RGB)

plt.subplot(121)
plt.title(f'J1 image')
plt.imshow(J1)

# (c) make new color
imgJ2 = imgJ1
imgJ2[:, :, 0] = imgJ1[:, :, 2]
imgJ2[:, :, 1] = imgJ1[:, :, 0]
imgJ2[:, :, 2] = imgJ1[:, :, 1]

# (d) show J2 image adn write out as a JPEG file
plt.subplot(122)
plt.title(f'J2 image')
plt.imshow(imgJ2)

cv.imwrite('HW23.jpeg', imgJ2)
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
