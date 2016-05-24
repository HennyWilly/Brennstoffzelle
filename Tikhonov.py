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
    
def optimal_tikhonov_parameter(A, b):
    """
    Hier koennte Ihr PythonDoc stehen
    """
    
    #Das ist offensichtlich nicht immer der optiale Wert, nur ein Platzhalter...
    return 1/10.0
    