from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser(description='Object labelling tool')
parser.add_argument('--input',
                    help='Image source')

args = parser.parse_args()

image_name = args.input
image = cv2.imread(image_name)

image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
# image = np.array(image)

show_crosshair = False
from_center = False
r = cv2.selectROI("Image", image, from_center, show_crosshair)
# Crop image
img_crop = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

# Display cropped image
print(img_crop.shape, type(img_crop))
#cv2.imshow("Image", imCrop)
plt.imshow(img_crop)
plt.show()

im = Image.fromarray(img_crop)
im.save("cropped.jpg")
