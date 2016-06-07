# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:29:10 2016

@author: studi2014
"""

from dolfin import *
import pylab as pl
import numpy as np
import matplotlib.pyplot as mpl
from BiotSavart import BiotSavartMatrix
from matplotlib import cm
from numpy.linalg import pinv
from numpy.linalg import svd

    
def Gradient():    
    # Create classes for defining parts of the boundaries and the interior
    # of the domain
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0)
    
    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1.0)
    
    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            #if(near(x[1], 0.0)) :
            #    return not near(x[0], 0.5)
            #return False
            return near(x[1], 0.0) and not near(x[0], 10.0/20)
            
    class BottomOld(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[1], 0.0))
    
    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 1.0)
    
    class Obstacle(SubDomain):
        def inside(self, x, on_boundary):
            return False
    
    
    class FunctionRight(Expression):
    
        def eval(self, values, x):
            if x[1] < 0.5 + DOLFIN_EPS:
                values[0] = 0
            else:
                values[0] = 5
        
    n=30
    h=1.0/n
    mesh = UnitSquareMesh(n, n)
    j= {}
    for i in range(n+1) :
        j[str(float(i)/float(n))] = 5
        
    print(mesh)
                
    # Initialize sub-domain instances
    left = Left()
    top = Top()
    right = Right()
    bottom1 = BottomOld()
    bottom = Bottom()
    obstacle = Obstacle()
    
    # Define mesh
    
    # Initialize mesh function for interior domains
    domains = CellFunction("size_t", mesh)
    domains.set_all(0)
    obstacle.mark(domains, 1)
    
    # Initialize mesh function for boundary domains
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    top.mark(boundaries, 2)
    right.mark(boundaries, 3)
    bottom1.mark(boundaries, 5)
    bottom.mark(boundaries, 4)
    
    # Define input data
    a0 = Constant(1.00)
    g_L = Constant("0")
    g_T = Constant("0")
    g_B = Constant("0")
    g_R = FunctionRight()
    f = Constant(0.0)
    
    # Define function space and basis functions
    V = FunctionSpace(mesh, "CG", 2)
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # Define Dirichlet boundary conditions at top and bottom boundaries
    bcs = [DirichletBC(V, 0.0, boundaries, 5)]
    
    # Define new measures associated with the interior domains and
    # exterior boundaries
    dx = Measure("dx")[domains]
    ds = Measure("ds")[boundaries]
    
    # Define variational form
    F = (inner(a0*grad(u), grad(v))*dx 
         - g_L*v*ds(1) - g_T*v*ds(2) - g_R*v*ds(3) - g_B*v*ds(4) - f*v*dx)
    
    # Separate left and right hand sides of equation
    a, L = lhs(F), rhs(F)
    
    # Solve problem
    u = Function(V)
    solve(a == L, u, bcs)
        
    # Plot solution and gradient
    #plot(u, title="u")
    gul=grad(u)
    plot(gul, title="Projected grad(u) left")
    gur=grad(-u)
    plot(gur, title="Projected grad(-u) right")
    #gu=np.concatenate((gul,gur),axis=0)

    
    prol=project(gul)
    pror=project(gur)
    
    def gradient(x,y):
        if (x>0):
            return(pror(1-x,y))
        return(prol(1+x,y))
        
        
        
    def BiotSavart2(x) :
        
        if(x[0] < 1 and x[0] > -1 and x[1] > 0 and x[1] < 1) :
            return 0
            
        H=0
        for yi in range(n):
            for xi in range(n*2):
                xleft=-1+0.5*h+xi*h
                y=0.5*h+yi*h
                
                gra=gradient(xleft,y)
                
                x0=x[0]-xleft
                x1=x[1]-y
                H += (gra[0]*x1 - gra[1]*x0)/(sqrt(x0*x0+x1*x1)**3) 
                
        return H / (4*np.pi)
                
                
    return(gradient, BiotSavartMatrix(gradient))
    #flux = [1,1]*grad(u)
    
    #plot(flux, title='flux field')
    
    #flux_x, flux_y = flux.split(deepcopy=True)  # extract components
    #plot(flux_x, title='x-component of flux (-p*grad(u))')
    #plot(flux_y, title='y-component of flux (-p*grad(u))')
    
def test() :
    (g, o)=Gradient()
    ynorm=np.zeros((201,101))
    yin0=np.zeros((201,101))
    yin1=np.zeros((201,101))
    
    
    x=range(0,101)
    x = np.array(x)
    x = x/100.
    y = np.array(x)
    y2 = np.array(x)
    for i in range(101) :
        y[i]=g(0,x[i])[0] 
        y2[i]=g(0,x[i])[1]
    mpl.figure()    
    mpl.plot(x,y)
    mpl.plot(x,y2)
        
    for i in range(201) :
        for j in range(101) :
            cx=(i/100.0) - 1
            cy=(j/100.0)
            ynorm[i,j]=np.linalg.norm(g(cx,cy))
            yin0[i,j]=g(cx,cy)[0]
            yin1[i,j]=g(cx,cy)[1]
            
    Hp=np.zeros((40,20))
    for i in range(40) :
        for j in range(20) :
            cx=(i/10.0) -2
            cy=(j/5.0)-1
            Hp[i,j]=o((cx,cy))
                
    
    mpl.figure()
    mpl.imshow(ynorm)
    mpl.colorbar()
    
    mpl.figure()
    mpl.imshow(yin0)
    mpl.colorbar()
    
    mpl.figure()
    mpl.imshow(yin1)
    mpl.colorbar()
    
    
    mpl.figure()
    mpl.imshow(Hp, cmap=cm.coolwarm)
    mpl.colorbar()
    
    mpl.show()
    
    
(g, v) = Gradient()
zz=np.dot(v[0],v[1])
Ai = pinv(v[0], rcond=10E-10)
v_erg = np.dot(Ai, zz)

mpl.figure()
mpl.plot(v[1][0::2,:])
mpl.plot(v_erg[0::2,:])


x=range(0,101)
x = np.array(x)
x = x/100.
y = np.array(x)
y2 = np.array(x)
for i in range(101) :
    y[i]=g(0,x[i])[0] 
    y2[i]=g(0,x[i])[1]
#mpl.figure()    
#mpl.plot(x,y)
#mpl.plot(x,y2)
print(v[3])
mpl.show()