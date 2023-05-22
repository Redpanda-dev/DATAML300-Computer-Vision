import cv2
import numpy as np

img = cv2.imread("reference.jpg", cv2.IMREAD_GRAYSCALE)  # query image
test_img = cv2.imread("image.jpg") # test image

# Features
sift = cv2.SIFT_create()
keyp_image, descrip_image = sift.detectAndCompute(img, None)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Convert the  test image to grayscale using proper cv2-function.
# After that calculate the keypoints and descriptors with SIFT.
# Then calculate the matches between both query and test image descriptors
# with already declared flann using knnMatch-function (k = 2). 
# Store the matches to "matches"-variable.

##--your-code-starts-here--##
grayframe = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) # replaced
keyp_grayframe, descrip_keyframe = sift.detectAndCompute(grayframe, None) # replaced
matches = flann.knnMatch(descrip_image, descrip_keyframe, k=2) # replaced
##--your-code-ends-here--##

good_points = []
thresh = 0.6

for m, n in matches:
    if m.distance < thresh * n.distance:
        good_points.append(m)
        
# Visualize matches # Uncomment to see matches visualized. Might not work depending on how you calculated the matches.
img_matches = np.empty((max(img.shape[0], test_img.shape[0]), img.shape[1]+test_img.shape[1], 3), dtype=np.uint8)
cv2.drawMatches(img, keyp_image, test_img, keyp_grayframe, 
                 good_points, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

 # Label left image as query image and right image as test image in the lower left corner
cv2.putText(img_matches, "Query Image", (10, img_matches.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
cv2.putText(img_matches, "Test Image", (img.shape[1] + 10, img_matches.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
cv2.imshow('Good Matches', img_matches)
        
cv2.imshow("Query image", img)


if len(good_points) > 20:
    query_pts = np.float32([keyp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    test_pts = np.float32([keyp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
    
    # Calculate the homography using cv2.findHomography, look up the documentation
    # (https://docs.opencv.org/master/d9/d0c/group__calib3d.html)
    # for the function to see what values it takes in. Store this homography matrix to 
    # variable "matrix". Note that the function returns the mask as well and 
    # the code will throw an error if you don't store it anywhere. 
    
    ##--your-code-starts-here--##
    matrix, _ = cv2.findHomography(query_pts, test_pts, cv2.RANSAC,5.0)  # replace me
    ##--your-code-ends-here--##
    
    # Perspective transform
    h, w = img.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, matrix)
    homography = cv2.polylines(test_img, [np.int32(dst)], True, (255, 0, 0), 3)
    cv2.imshow("Homography", homography)    
    
    # Warp the image using cv2.warpPerspective and the homography matrix 
    # so the target is in one to one correspondence to query image
    # in terms of perspective.
    # Use dsize = (720, 540)
    # HINT: In order to produce the inverse of what the homography does what 
    # should you do with the homography matrix?
    
    ##--your-code-starts-here--##
    im_warped = cv2.warpPerspective(test_img, np.linalg.inv(matrix), (720, 540)) # replace me
    ##--your-code-ends-here--##
    cv2.imshow("Warped image", im_warped)

else:
    cv2.imshow("Homography", grayframe)

cv2.waitKey(0)