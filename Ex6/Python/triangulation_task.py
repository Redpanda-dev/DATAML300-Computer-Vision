import numpy as np
import matplotlib.pyplot as plt
from trianglin import trianglin


# Visualization of three point correspondences in both images
im1 = plt.imread('im1.jpg')
im2 = plt.imread('im2.jpg')

# Points L, M, N (corners of the book) in image 1
lmn1 = 1.0e+03 * np.array([[1.3715, 1.0775], 
                           [1.8675, 1.0575],
                           [1.3835, 1.4415]])

# Points L, M, N (corners of the book) in image 2
lmn2 = 1.0e+03 * np.array([[1.1555, 1.0335],
                           [1.6595, 1.0255],
                           [1.1755, 1.3975]])

# Annotate and show images
labels = ['L', 'M', 'N']
plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(im1)
for i in range(len(labels)):
    plt.plot(lmn1[i, 0], lmn1[i, 1], 'c+', markersize=10)
    plt.annotate(labels[i], (lmn1[i, 0], lmn1[i, 1]), color='c', fontsize=20)
plt.subplot(2,1,2)
plt.imshow(im2)
for i in range(len(labels)):
    plt.plot(lmn2[i, 0], lmn2[i, 1], 'c+', markersize=10)
    plt.annotate(labels[i], (lmn2[i, 0], lmn2[i, 1]), color='c', fontsize=20)
plt.xticks([])
plt.yticks([])


# Your task is to implement the missing function 'trianglin.py'
# The algorithm is described in the lecture slides and exercise sheet.
# Output should be the homogeneous coordinates of the triangulated point.

# Load the pre-calculated projection matrices
P1 = np.load('P1.npy')
P2 = np.load('P2.npy')

# Triangulate each corner
L = trianglin(P1, P2,
              np.hstack((lmn1[0, :].T, [1])),
              np.hstack((lmn2[0, :].T, [1])))
M = trianglin(P1, P2,
              np.hstack((lmn1[1, :].T, [1])),
              np.hstack((lmn2[1, :].T, [1])))
N = trianglin(P1, P2,
              np.hstack((lmn1[2, :].T, [1])),
              np.hstack((lmn2[2, :].T, [1])))
                      
# We can then compute the width and height of the picture on the book cover
# Convert the above points to cartesian, form vectors corresponding to
# book covers horizontal and vertical sides using the points and calculate 
# the norm of these to acquire the height and width (mm).
##-your-code-starts-here-##
# Convert the triangulated points to cartesian coordinates
L = L[:3] / L[-1]
M = M[:3] / M[-1]
N = N[:3] / N[-1]

# Compute the vectors corresponding to the book cover's horizontal and vertical sides
u = M - L
v = N - L
 
picture_w_mm = np.linalg.norm(u)
picture_h_mm = np.linalg.norm(v)
##-your-code-ends-here-##
print("Picture width: %.2f mm" % picture_w_mm)
print("Picture height: %.2f mm" % picture_h_mm)
plt.show()
