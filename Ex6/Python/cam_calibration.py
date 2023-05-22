import numpy as np
import matplotlib.pyplot as plt
from camcalibDLT import camcalibDLT
from string import ascii_lowercase


# The given image coordinates were originally localized manually
im1 = plt.imread('im1.jpg')

x1 = 1.0e+03 * np.array([0.7435, 3.3315, 0.8275, 3.2835, 
                         0.5475, 3.9875, 0.6715, 3.8835])
    
y1 = 1.0e+03 * np.array([0.4455, 0.4335, 1.7215, 1.5615, 
                         0.3895, 0.3895, 2.1415, 1.8735])
                      
# Image coordinates of points as rows of matrix 'abcdefgh'
abcdefgh = np.vstack((x1, y1)).T

# World coordinates of the points (dimensions of the shelf)
ABCDEFGH_w = np.array([[758, 0, -295],
                       [0, 0, -295],
                       [758, 360, -295],
                       [0, 360, -295],
                       [758, 0, 0],
                       [0, 0, 0],
                       [758, 360, 0],
                       [0, 360, 0]])

# Plot manually localized points
plt.figure(1)
plt.title('Cyan: manually localized points   Red: projected points')
plt.imshow(im1)
plt.plot(x1, y1, 'c+', markersize=10)
for i in range(len(x1)):    
    plt.annotate(ascii_lowercase[i], (x1[i], y1[i]), color='c', fontsize=20)

# Your task is to implement the missing function camcalibDLT.py.
# The algorithm is summarised on the lecture slides and exercise sheet.
# The function takes the homogeneous coordinates of the points as input
P1 = camcalibDLT(np.hstack((ABCDEFGH_w, np.ones((8, 1)))),
                 np.hstack((abcdefgh,   np.ones((8, 1)))))

# Check the results by projecting the world points with the estimated P.
# The projected points should overlap with manually localized points
pproj1 = np.dot(P1, np.hstack((ABCDEFGH_w, np.ones((8, 1)))).T)
for i in range(pproj1.shape[1]):
    plt.plot(pproj1[0, i] / pproj1[2, i],
             pproj1[1, i] / pproj1[2, i],
             'rx', markersize=12)
plt.xticks([])
plt.yticks([])


# Calibration of the second camera
im2 = plt.imread('im2.jpg')

x2 = 1.0e+03 * np.array([0.5835, 3.2515, 0.6515, 3.1995, 
                         0.1275, 3.7475, 0.2475, 3.6635])
    
y2 = 1.0e+03 * np.array([0.4135, 0.4015, 1.6655, 1.5975, 
                         0.3215, 0.3135, 2.0295, 1.9335])
                     
plt.figure(2)
plt.title('Cyan: manually localized points   Red: projected points')
plt.imshow(im2)
plt.plot(x2, y2, 'c+', markersize=10)
for i in range(len(x1)):    
    plt.annotate(ascii_lowercase[i],
                 (x2[i], y2[i]),
                 color='c', fontsize=20)

# Second camera projection matrix
P2 = camcalibDLT(np.hstack((ABCDEFGH_w, np.ones((8, 1)))),
                 np.vstack((x2, y2, np.ones(8))).T)

# Again, check results by projecting world coordinates with P2
pproj2 = np.dot(P2, np.hstack((ABCDEFGH_w, np.ones((8, 1)))).T)
for i in range(pproj2.shape[1]):
    plt.plot(pproj2[0, i] / pproj2[2, i],
             pproj2[1, i] / pproj2[2, i],
             'rx', markersize=12)
plt.xticks([])
plt.yticks([])
plt.show()
