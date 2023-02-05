#from pylab import *
import numpy as np

def linefitlsq(x,y):
    xm = np.mean(x)
    ym = np.mean(y)
    
    U = np.vstack((x-xm, y-ym)).T
    W, V = np.linalg.eig(np.dot(U.T, U))
    minv = np.amin(W)
    minid = np.argmin(W)
    ev = -V[:,minid]
    
    a = ev[0]
    b = ev[1]
    d = a*xm + b*ym
    l = np.array([a,b,-d])
    
    return l
    