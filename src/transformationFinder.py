import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

MIN_MATCHES = 10

def create_parser():
    parser =argparse.ArgumentParser()
    parser.add_argument('--left_image', type=str, required=True)
    parser.add_argument('--right_image', type=str, required=True)

    return parser

def read_images(opts, visualize=False):
    image_with_fruit = cv2.imread(opts.left_image)
    image_with_fruit = cv2.cvtColor(image_with_fruit, cv2.COLOR_BGR2RGB)
    image_without_fruit = cv2.imread(opts.right_image)
    image_without_fruit = cv2.cvtColor(image_without_fruit, cv2.COLOR_BGR2RGB)

    if visualize:
        plt.imshow(image_with_fruit)
        plt.imshow(image_without_fruit)
        plt.show()

    return image_with_fruit, image_without_fruit

def SIFT_match(img1, img2, min_matches=MIN_MATCHES):
    sift = cv2.xfeatures2d_SIFT.create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    valid_matches = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            valid_matches.append(m)

    return valid_matches, kp1, kp2

def find_homography(matches, kp1, kp2, min_matches=MIN_MATCHES):
    if len(matches) > min_matches:
        src_pts = np.float32(
            [kp1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        return H, mask, matchesMask

    else:
        print("Not enough matches are found - %d/%d" %
              (len(matches), min_matches))
        matchesMask = None
    
    return None, None, None

def draw_matches(img1, img2, kp1, kp2, matches, matchesMask):
    # Initiate SIFT detector
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    image_with_matches = cv2.drawMatches(img1, kp1, img2,
                                         kp2, matches, None, **draw_params)

    return image_with_matches     

def main(opts):

    image_with_fruit, image_without_fruit = read_images(opts)

    matches, kp1, kp2 = SIFT_match(image_with_fruit, image_without_fruit)    

    H, mask, matchesMask = find_homography(matches, kp1, kp2)

    h, w, _ = image_with_fruit.shape
    
    # Hard Coded values for cropped box
    y1 = 405*2
    y2 = 480*2
    x1 = 277*2
    x2 = 410*2

    fruit_crop = np.uint8(np.zeros((h, w, 3)))
    fruit_crop[y1:y2, x1:x2] = image_with_fruit[y1:y2, x1:x2]

    fruit_crop_warped = cv2.warpPerspective(fruit_crop, H, (1920, 1080))

    image_with_fruit_added = image_without_fruit.copy()

    image_with_fruit_added[fruit_crop_warped != 0] = fruit_crop_warped[fruit_crop_warped != 0]
    # plt.imshow(image_with_fruit_added)
    # plt.show()

    image_with_fruit_added_pers1 = cv2.warpPerspective(
        image_with_fruit_added, np.linalg.inv(H), (1080, 1920))
    plt.imshow(image_with_fruit_added_pers1)
    plt.show()

if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    main(opts)
