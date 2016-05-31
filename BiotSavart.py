# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:55:35 2016

@author: studi2014
"""

from pylab import *
from numpy.linalg import norm

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
            
def oldBiotSavartMatrix(x, jl, jr):
    n = 100
    h = 1. / n
    Kl = lambda x,y : cross(jl(y), (x-y)/norm(x - y)**3)
    Kr = lambda x,y : cross(jr(y), (x-y)/norm(x - y)**3)
    L = zeros((n/2, n), float)
    R = zeros((n/2, n), float)
    for j in range(n/2):
        for i in range(1, n - 1):
            y = array([i, j])
            L[i][j] = 2 * Kl(x, y)
            y = array([i + 0.5, j])
            R[i][j] = 2 * Kr(x, y)
        y = array([0, j])
        L[0][j] = Kl(x, y)
        y = array([1, j])
        L[n-1][j] = Kl(x, y)
        y = array([0.5, j])
        R[0][j] = Kr(x, y)
        y = array([i + 0.5, 1])
        R[n-1][j] = Kr(x, y)
            
    return h/(8*math.pi) * (L + R)
    
def BiotSavartMatrix(x, jl):
    n = 100
    m = n / 4
    h = 1. / n
    Kl = lambda x,y : cross(jl(y), (x-y)/norm(x - y)**3)
    M = zeros((n, n), float)
    for (i, y) in enumerate(itertools.product(range(1, m-1), range(2))):
        y = array(y)
        M[i][0] = 2 * Kl(x, y)
    y = array([0, j])
    M[0][0] = Kl(x, y)
    y = array([1, j])
    M[n-1][0] = Kl(x, y)
            
    return h/(8*math.pi) * M