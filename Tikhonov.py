# -*- coding: utf-8 -*-

from pylab import *
from scipy.linalg import *
from scipy.integrate import *
import numpy as np
import numpy.matlib as matlib
import scipy.optimize as spOpt
from scipy.linalg import *

def tikhonov(A, b, alpha, allowNegative=True):
    """
    Wendet die Tikhonov-Regularisierung auf die Matrix A und die Lösung b an.
    
    @type  A: matrix
    @param A: Die zu regularisierende Matrix.
    @type  b: vector
    @param b: Die Lösung der Matrixoperation.
    @type  alpha: number
    @param alpha: Der Regularisierungsparameter.
    @type  allowNegative: boolean 
    @param allowNegative: Wenn false, werden nur positive x-Werte zur Annäherung erlaubt; sonst auch negative.
    
    @rtype: vector, number, number
    @return: Solution, Residuum, Norm der Lösung
    """
    
    n = A.shape[0]
    m = A.shape[1]
    A1 = np.concatenate((A, alpha * matlib.identity(m)))
    b1 = np.concatenate((b, np.zeros(shape=(n,1))))
    
    print A, A.shape
    print A1, A1.shape

    if (allowNegative):
        x, res, rank, s = lstsq(A1, np.squeeze(b1))
        return x, res, norm(x)
    else:
        x, res = spOpt.nnls(A1, np.squeeze(b1))
        return x, res, norm(x)
    
def tikhonov2(A, b, alpha):
    n = np.amax(A.shape)
    
    part1 = np.dot(A.transpose(), A)
    part2 = alpha * np.identity(n)
    part3 = np.add(part1, part2)
    part4 = inv(part3)
    part5 = np.dot(part4, A.transpose())
    part6 = np.dot(part5, b)
    
    return part6
    
def l_curve(A, b, n):
    """
    Hier koennte Ihr PythonDoc stehen
    """
    tt = linspace(0,1,n)
    
    resArray = np.zeros(n)
    normArray = np.zeros(n)
    alphaArray = np.zeros(n)
    
    for (i, t) in enumerate(tt):
        x, res, norm = tikhonov(A, b, t) 
        if res:     
            resArray[i] = res
            normArray[i] = norm
            alphaArray[i] = t
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    
    plt.plot(resArray, normArray, picker=2)
    plt.loglog()
    ax.set_xlabel("Residuum")
    ax.set_ylabel("Norm")
    def onpick1(event):
        if isinstance(event.artist, Line2D):
            print('alpha(s):', alphaArray[event.ind])

    fig.canvas.mpl_connect('pick_event', onpick1)
    plt.show()

def noisy_y(y, delta):
    n, m = y.shape
    z = np.random.randn(n, 1)
    
    return array( [ y[i] + delta * z[i] for i in xrange(0, n)] )