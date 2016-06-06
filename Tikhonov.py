# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib as matlib
import scipy.optimize as spOpt
from scipy.linalg import *

def tikhonov(A, b, alpha, allowNegative):
    """
    Hier koennte Ihr PythonDoc stehen
    """
    
    n = A.shape[0]
    A1 = np.concatenate((A, alpha * matlib.identity(n)))
    b1 = np.concatenate((b, np.zeros(shape=(n,1))))

    if (allowNegative):
        return lstsq(A1, np.squeeze(b1))[0]
    else:
        return spOpt.nnls(A1, np.squeeze(b1))[0]
    
def l_curve(A, b, n):
    """
    Hier koennte Ihr PythonDoc stehen
    """
    tt = linspace(0,1,n)
    
    resArray = np.zeros(n)
    normArray = np.zeros(n)
    alphaArray = np.zeros(n)
    
    i = 0
    for t in tt:
        x, res, norm = tkhnv.tikhonov(A, b, t) 
        if res:     
            resArray[i] = res
            normArray[i] = norm
            alphaArray[i] = t
        i = i + 1
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    plt.plot(resArray, normArray, 'ro')
    for j in range(0, n, 10):
        ax.annotate('(%s)' % alphaArray[j], xy=(resArray[j], normArray[j]), textcoords='data')
    plt.loglog()
    plt.show()