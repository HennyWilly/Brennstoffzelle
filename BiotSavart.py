# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:55:35 2016

@author: studi2014
"""

from pylab import *
from numpy.linalg import norm
from math import pi
import itertools

def BiotSavart(x, J):
    n = 100
    h = 0.5 / n
    intl = 0
    intr = 0
    #oben und unten
    tt = linspace(-1 + h, 1 - h, n)
    for j in range(2):        
        for i in tt:
            y = array([i, j])
            intl += 2 * cross(J(y), (x-y)/norm(x - y)**3)
            y = array([i + 1, j])
            intr += 2 * cross(J(y), (x-y)/norm(x - y)**3)
        y = array([-1, j])
        intl += cross(J(y), (x-y)/norm(x - y)**3)
        y = array([0, j])
        intr += cross(J(y), (x-y)/norm(x - y)**3)
        y = array([0, j])
        intl += cross(J(y), (x-y)/norm(x - y)**3)
        y = array([1, j])
        intr += cross(J(y), (x-y)/norm(x - y)**3)
            
    h = 1. / n
    #Seiten der Elemente
    tt = linspace(0 + h, 1 - h, n)
    for i in (-1, 0):    
        for j in tt:
            y = array([i, j])
            intl += 2 * cross(J(y), (x-y)/norm(x - y)**3)
            y = array([i + 1, j])
            intr += 2 * cross(J(y), (x-y)/norm(x - y)**3)
        y = array([i, 0])
        intl += cross(J(y), (x-y)/norm(x - y)**3)
        y = array([i + 1, 0])
        intr += cross(J(y), (x-y)/norm(x - y)**3)
        y = array([i, 1])
        intl += cross(J(y), (x-y)/norm(x - y)**3)
        y = array([i + 1, 1])
        intr += cross(J(y), (x-y)/norm(x - y)**3)
    
    return 1 / (4 * math.pi) * (intl + intr)
    
def BiotSavartMatrix(J, n=21*21., m=6**2):
    #n Auswwertungsstellen
    #m Messstellen
    #2n > m, sonst liegt ein unterbestimmtes System vor
    BS = zeros((m, 2 * n), float)
    discJ = zeros((2 * n, 1), float)
    yList = zeros((n, 2), float)
    xList = zeros((m, 2), float)
    for (i, y) in enumerate(itertools.product(linspace(-1, 1, sqrt(n)), linspace(0, 1, sqrt(n)))):
        #diskrete Auswertungsstellen
        yList[i] = array(y)
        
    for (i, x) in enumerate(itertools.chain(itertools.product(linspace(-1.1, 1.1, m/4.), (-0.1, 1.1)), itertools.product((-1.1, 1.1), linspace(-0.1, 1.1, m/4.)))):
        #diskrete Messstellen
        xList[i] = array(x)     
        
    for (i, y) in enumerate(yList):    
        for (j, x) in enumerate(xList):
            #Matrix aufstellen
            BS[j][2*i] = BS[j][2*i] + 1/(4*pi*n) * (x[1] - y[1]) / norm(x - y)**3
            BS[j][2*i + 1] = BS[j][2*i + 1] + 1/(4*pi*n) * (x[0] - y[0]) / norm(x - y)**3
            if (y[0] == -1 or y[0] == 1):
                BS[j][2*i] = BS[j][2*i] / 2
                BS[j][2*i + 1] = BS[j][2*i + 1] / 2
                
            if (y[1] == 0 or y[1] == 1):
                BS[j][2*i] = BS[j][2*i] / 2
                BS[j][2*i + 1] = BS[j][2*i + 1] / 2
            
        #diskrete J's
        discJ[2*i] = J(*y)[0]
        discJ[2*i + 1] = J(*y)[1]
            
    return BS, discJ, xList, yList
    
if __name__ == '__main__':
    BS, J, x = BiotSavartMatrix(lambda x: (1, 1))
    print BS, J, x