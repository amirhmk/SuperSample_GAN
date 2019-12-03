from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_with_fruit = cv2.imread("./images/whiteleft3.jpg")
image_with_fruit = cv2.cvtColor(image_with_fruit, cv2.COLOR_BGR2RGB)
image_without = cv2.imread("./images/whiteright3.jpg")
image_without = cv2.cvtColor(image_without, cv2.COLOR_BGR2RGB)

img1 = cv2.imread("./images/whiteleft3.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("./images/whiteright3.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# plt.imshow(img1)
# plt.show()
plt.imshow(img2)
plt.show()

MIN_MATCH_COUNT = 10

# Initiate SIFT detector
sift = cv2.xfeatures2d_SIFT.create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w,z = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

# plt.imshow(img3, 'gray'),plt.show()
# Sort matches in the order of their distance.
# matches = sorted(matches, key=lambda x: x.distance)
#
# draw_matches = np.array([[]])
# draw_matches = cv2.drawMatches(image_with_fruit, kp1, image_without, kp2, matches[:20], draw_matches, flags=2, matchColor=[255, 255, 0])
#
# plt.imshow(draw_matches)
# plt.show()

# plt.subplot(121),plt.imshow(cv2.warpPerspective(img1, M, (1750, 1250)))
# plt.subplot(122),plt.imshow(img2)
# plt.show()

# 405 480 277 410

# Hard Coded values for cropped box
y1 = 405*2
y2 = 480*2
x1 = 277*2
x2 = 410*2


print(img1[405*2:480*2, 277*2:410*2])
fruit_crop = np.uint8(np.zeros((img1.shape[0], img1.shape[1], 3)))
fruit_crop[405*2:480*2, 277*2:410*2] = img1[405*2:480*2, 277*2:410*2]

fruit_crop_warped = cv2.warpPerspective(fruit_crop, M, (1920, 1080))
# plt.imshow(fruit_crop_warped)
# plt.show()


fruit_grey = cv2.cvtColor(fruit_crop_warped, cv2.COLOR_RGB2GRAY)

img2[fruit_grey != 0] = fruit_crop_warped[fruit_grey != 0]
plt.imshow(img2)
plt.show()
