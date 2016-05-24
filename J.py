# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:29:10 2016

@author: studi2014
"""

from dolfin import *
import numpy as np
import matplotlib.pyplot as mpl
    
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
            if x[1] < 0.25 + DOLFIN_EPS:
                values[0] = 5
            elif x[1] < 0.5 + DOLFIN_EPS:
                values[0] = 0
            else:
                values[0] = 5
        
    n=20
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
    
    # Evaluate integral of normal gradient over top boundary
    n = FacetNormal(mesh)
    m1 = dot(grad(u), n)*ds(2)
    v1 = assemble(m1)
    #print "\int grad(u) * n ds(2) = ", v1
    
    # Evaluate integral of u over the obstacle
    m2 = u*dx(1)
    v2 = assemble(m2)
    #print "\int u dx(1) = ", v2
    
    # Plot solution and gradient
    #plot(u, title="u")
    gul=grad(u)
    #plot(gul, title="Projected grad(u) left")
    gur=grad(-u)
    #plot(gur, title="Projected grad(-u) right")
    #gu=np.concatenate((gul,gur),axis=0)

    
    prol=project(gul)
    pror=project(gur)
    
    def gradient(x,y):
        if (x>0):
            return(pror(x,y))
        return(prol(x+1,y))
        
    
    return(gradient)
    #flux = [1,1]*grad(u)
    
    #plot(flux, title='flux field')
    
    #flux_x, flux_y = flux.split(deepcopy=True)  # extract components
    #plot(flux_x, title='x-component of flux (-p*grad(u))')
    #plot(flux_y, title='y-component of flux (-p*grad(u))')
    
    

     