from PIL import Image
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


# Load images
img1 = np.array(Image.open('boat1.png'))
img2 = np.array(Image.open('boat6.png'))

# Pre-calculated SURF-feature vectors f1 and f2
# This can be done with OpenCV or Matlab's built-in functions
f1 = loadmat('f1.mat')['f1']
f2 = loadmat('f2.mat')['f2']

# SURF-points' location and scale
vpts1_loc = loadmat('vpts1.mat')['vpts1_loc']
vpts1_scale = loadmat('vpts1.mat')['vpts1_scale']
vpts2_loc = loadmat('vpts2.mat')['vpts2_loc']
vpts2_scale = loadmat('vpts2.mat')['vpts2_scale']

# Compute the pairwise distances of feature vectors to matrix 'distmat'
distmat = np.dot(f1, f2.T)
X_terms = np.expand_dims(np.diag(np.dot(f1, f1.T)), axis=1)
X_terms = np.tile(X_terms,(1,f2.shape[0]))
Y_terms = np.expand_dims(np.diag(np.dot(f2, f2.T)), axis=0)
Y_terms = np.tile(Y_terms,(f1.shape[0],1))
distmat = np.sqrt(Y_terms + X_terms - 2*distmat)

# Determine the mutually nearest neighbors
dist1 = np.amin(distmat, axis=1)
ids1 = np.argmin(distmat, axis=1)
dist2 = np.amin(distmat, axis=0)
ids2 = np.argmin(distmat, axis=0)

pairs = []
for k in range(ids1.size):
    if k == ids2[ids1[k]]:
        pairs.append(np.array([k, ids1[k], dist1[k]]))
pairs = np.array(pairs)


################### Nearest neighbor based sorting ###########################
# Sort the mutually nearest neighbors based on the distance
snnd = np.sort(pairs[:,2], axis=0)
id_nnd = np.argsort(pairs[:,2], axis=0)

# Visualize the 5 best matches
Nvis = 5

plt.figure(figsize=(16, 8))
plt.suptitle("Top 5 mutual nearest neigbors of SURF features", fontsize=20)
plt.imshow(np.hstack((img1, img2)), cmap='gray')

t = np.arange(0, 2*np.pi, 0.1)

# Display matches
for k in range(Nvis):
    pid1 = pairs[id_nnd[k], 0]
    pid2 = pairs[id_nnd[k], 1]
        
    loc1 = vpts1_loc[int(pid1)]
    r1 = 6*vpts1_scale[int(pid1)]
    loc2 = vpts2_loc[int(pid2)]
    r2 = 6*vpts2_scale[int(pid2)]
    
    plt.plot(loc1[0]+r1*np.cos(t), loc1[1]+r1*np.sin(t), 'm-', linewidth=3)
    plt.plot(loc2[0]+r2*np.cos(t)+img1.shape[1], loc2[1]+r2*np.sin(t), 'm-', linewidth=3)
    plt.plot([loc1[0], loc2[0]+img1.shape[1]], [loc1[1], loc2[1]], 'c-')
    
# How many of the top 5 matches appear to be correct correspondences?

    
################## Nearest neighbor ratio based sorting #######################
# Now, your task is to compute and visualize the top 5 matches based on
# the nearest neighbor distance ratio defined in exercise sheet.
# How many of those are correct correspondences?
#
# HINT:  Loop through the first column in 'pairs' (first feature vector indices),
# use each index value from this column to get the corresponding row from
# distmat_sorted, get the nearest and second nearest distances from this row
# to calculate the distance ratio and store this to nndr.
# Remember to sort and save the sorted indices to id_nndr similarly as above.

distmat_sorted = np.sort(distmat, axis=1)  # each row sorted in ascending order
nndr=np.zeros(pairs.shape[0])  # pre-allocate memory

##-your-code-starts-here-##

##-your-code-ends-here-##

# Visualize the 5 best matches
Nvis = 5

plt.figure(figsize=(16, 8))
plt.suptitle("SURF matching with NNDR", fontsize=20)
plt.imshow(np.hstack((img1, img2)), cmap='gray')
plt.title('Top 5 mutual nearest neighbors of SURF features')

# Display matches
t = np.arange(0, 2*np.pi, 0.1)
for k in range(Nvis):
    pid1 = pairs[id_nndr[k], 0]
    pid2 = pairs[id_nndr[k], 1]
    
    loc1 = vpts1_loc[int(pid1)]
    r1 = 6*vpts1_scale[int(pid1)]
    loc2 = vpts2_loc[int(pid2)]
    r2 = 6*vpts2_scale[int(pid2)]
    
    plt.plot(loc1[0]+r1*np.cos(t), loc1[1]+r1*np.sin(t), 'm-', linewidth=3)
    plt.plot(loc2[0]+r2*np.cos(t)+img1.shape[1], loc2[1]+r2*np.sin(t), 'm-', linewidth=3)
    plt.plot([loc1[0], loc2[0]+img1.shape[1]], [loc1[1], loc2[1]], 'c-')
plt.show()

# How many of the top 5 matches appear to be correct correspondences?