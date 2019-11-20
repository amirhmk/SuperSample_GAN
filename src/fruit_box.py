from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = Image.open("platefruits.jpg")
image = np.array(image)

showCrosshair = False
fromCenter = False
r = cv2.selectROI("Image", image, fromCenter, showCrosshair)
# Crop image
imCrop = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

# Display cropped image
print(imCrop.shape, type(imCrop))
#cv2.imshow("Image", imCrop)
plt.imshow(imCrop)
plt.show()

im = Image.fromarray(imCrop)
im.save("cropped.jpeg")
