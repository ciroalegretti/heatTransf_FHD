#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:35:54 2021

@author: alegretti
"""
from numba import jit,prange
import numpy as np
from pre import idNeighbors

@jit(nopython=True, parallel=True)
def poisson_cgs(volnumb,volarr,cv,h_data,dt,tol=1e-6):

    """Pseudo-transient poisson equation"""

    # h_data = bc_uvPsi.bc_uvPsi(h_data,volarr,cv,edge,S)
    Psi_o = h_data[:,3].copy()
    # psiOld = Psi_o.copy()
    W = h_data[:,4].copy()
    b = np.zeros((len(cv)))
    r = np.zeros((len(cv)))
    p = np.zeros((len(cv)))
    q = np.zeros((len(cv)))
    u = np.zeros((len(cv)))
    v = np.zeros((len(cv)))
    Au = np.zeros((len(cv)))
    Aq = np.zeros((len(cv)))

    for i in prange(volnumb):
        b[i] = W[i] + Psi_o[i]/dt
    # b = - h_data[:,4] + h_data[:,3]/dt

    l2_norm = 1.0

    it_number = 0
    it_max = len(cv)

    for i in prange(volnumb):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        W,N,E,S = idNeighbors.idNeighbors(volarr,i)

        r[i] = b[i] - (1/dt + (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*Psi_o[i] \
                    - (-1/(dxV*(cv[E,2] - cv[i,2])))*Psi_o[E]  \
                    - (-1/(dxV*(cv[i,2] - cv[W,2])))*Psi_o[W]  \
                    - (-1/(dyV*(cv[N,3] - cv[i,3])))*Psi_o[N]  \
                    - (-1/(dyV*(cv[i,3] - cv[S,3])))*Psi_o[S]                         

    r0 = r.copy()
    rho = 1#np.sum(r0*r)

    while it_number < it_max and l2_norm > tol:

        psik = Psi_o.copy()
        rk = r.copy()
        pk = p.copy()
        qk = q.copy()
        rhok = rho

        rho = np.sum(r0*r)
        beta = rho/rhok
        u = r + beta*qk
        p = u + beta*(qk + beta*pk)

        for i in prange(volnumb):
            dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
            dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
            W,N,E,S = idNeighbors.idNeighbors(volarr,i)

            v[i] = + (1/dt + (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*p[i] \
                    + (- 1/(dxV*(cv[E,2] - cv[i,2])))*p[E]  \
                    + (- 1/(dxV*(cv[i,2] - cv[W,2])))*p[W]  \
                    + (- 1/(dyV*(cv[N,3] - cv[i,3])))*p[N]  \
                    + (- 1/(dyV*(cv[i,3] - cv[S,3])))*p[S]   

        sigma = np.sum(r0*v)
        alpha = rho/sigma
        q = u - alpha*v

        for i in prange(volnumb):
            dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
            dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
            W,N,E,S = idNeighbors.idNeighbors(volarr,i)
            
            Au[i] = + (1/dt + (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*u[i] \
                    + (- 1/(dxV*(cv[E,2] - cv[i,2])))*u[E]  \
                    + (- 1/(dxV*(cv[i,2] - cv[W,2])))*u[W]  \
                    + (- 1/(dyV*(cv[N,3] - cv[i,3])))*u[N]  \
                    + (- 1/(dyV*(cv[i,3] - cv[S,3])))*u[S]                              
           
            Aq[i] = + (1/dt + (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*q[i] \
                    + (- 1/(dxV*(cv[E,2] - cv[i,2])))*q[E]  \
                    + (- 1/(dxV*(cv[i,2] - cv[W,2])))*q[W]  \
                    + (- 1/(dyV*(cv[N,3] - cv[i,3])))*q[N]  \
                    + (- 1/(dyV*(cv[i,3] - cv[S,3])))*q[S]  
        
        r = rk - alpha*(Au + Aq)
        Psi_o = psik + alpha*(u + q)
        
        l2_norm = np.sqrt(np.sum((r)**2))
        
        
        it_number += 1
        
        # print(l2_norm)
    
    h_data[:,3] = Psi_o

    return h_data

# @jit(nopython=True, parallel=True)
# def poisson2ldc_cgs(volnumb,volarr,cv,h_data,tol=1e-3):

#     """Fully implicit poisson equation"""

#     """PASSO 1"""
#     psiNew = h_data[:volnumb,3]#.astype('float128')#.copy()
#     b = np.zeros((volnumb))#.astype('float128')
#     src = - h_data[:volnumb,4]#.astype('float128')#np.zeros((volnumb)).astype('float128')
#     r = np.zeros((volnumb))#.astype('float128')
#     q = np.zeros((volnumb))#.astype('float128')
#     u = np.zeros((volnumb))#.astype('float128')
#     v = np.zeros((volnumb))#.astype('float128')
#     Au = np.zeros((volnumb))#.astype('float128')

#     # myfile = open('./0 -Results_MazumderEx3-1/{}_convPhi_{}vols_tol{:.1E}.dat'.format(tag,volnumb,tol),'w')
#     # myfile.write('Variables="it","R2"\n')
    
#     """STEP 2"""
#     # Construct b

#     for i in range(volnumb):
#         dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
#         dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
#         W,N,E,S = idNeighbors.idNeighbors(volarr,i)
#         """source terms Mazumder 3.1"""
        
#         bcLeft = 0.0
#         bcRight = 0.0#(1/4)*np.sinh(5) + (yH**2)*np.sinh(10*yH) + np.exp(2*cv[i,3])
#         bcBottom = 0.0#(1/4)*np.sinh(-5) + (xH**2)*np.sinh(10*xH) + 1
#         bcTop = 0.0#(1/4)*np.sinh(5) + (xH**2)*np.sinh(10*xH) + np.exp(2*cv[i,2])
        
#         """ b """
#         if W < 0:
#             b[i] = src[i] - 8*bcLeft/(3*dxV**2)
#         elif E < 0:
#             b[i] = src[i] - 8*bcRight/(3*dxV**2)
#         elif S < 0:
#             b[i] = src[i] - 8*bcBottom/(3*dyV**2)
#         elif N < 0:
#             b[i] = src[i] - 8*bcTop/(3*dyV**2)
#         elif W < 0 and N < 0:
#             b[i] = src[i] - (8/3)*(bcTop/dyV**2 + bcLeft/dxV**2)
#         elif W < 0 and S < 0:
#             b[i] = src[i] - (8/3)*(bcLeft/dxV**2 + bcBottom/dyV**2)
#         elif E < 0 and N < 0:
#             b[i] = src[i] - (8/3)*(bcRight/dxV**2 + bcTop/dyV**2)
#         elif E < 0 and S < 0:
#             b[i] = src[i] - (8/3)*(bcRight/dxV**2 + bcBottom/dyV**2)
#         else:
#             b[i] = src[i]
    
#     # Inicializando o resíduo
#     for i in range(volnumb):
#         dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
#         dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
#         W,N,E,S = idNeighbors.idNeighbors(volarr,i)
#         if W < 0:
#             r[i] = b[i] - (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                             - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*psiNew[i] \
#                         - ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
#                         - ( + 1/(dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  \
#                         - ( + 1/(dyV*(cv[i,3] - cv[S,3])))*psiNew[S]
#         elif E < 0:
#             r[i] = b[i] - (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                             - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*psiNew[i] \
#                         - ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
#                         - ( + 1/(dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  \
#                         - ( + 1/(dyV*(cv[i,3] - cv[S,3])))*psiNew[S]  
#         elif S < 0:
#             r[i] = b[i] - (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                             - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*psiNew[i] \
#                         - ( + 1/(dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
#                         - ( + 1/(dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
#                         - ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  
#         elif N < 0:
#             r[i] = b[i] - (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                             - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*psiNew[i] \
#                         - ( + 1/(dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
#                         - ( + 1/(dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
#                         - ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*psiNew[S]  
#         elif W < 0 and N < 0:
#             r[i] = b[i] - (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                             - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*psiNew[i] \
#                         - ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
#                         - ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*psiNew[S]  
#         elif W < 0 and S < 0:
#             r[i] = b[i] - (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                             - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*psiNew[i] \
#                         - ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
#                         - ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*psiNew[N]
#         elif E < 0 and N < 0:
#             r[i] = b[i] - (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                             - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*psiNew[i] \
#                         - ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
#                         - ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*psiNew[S]          
#         elif E < 0 and S < 0:
#             r[i] = b[i] - (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                             - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*psiNew[i] \
#                         - ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
#                         - ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  
#         else:
#             r[i] = b[i] - (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                             - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*psiNew[i] \
#                         - ( + 1/(dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
#                         - ( + 1/(dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
#                         - ( + 1/(dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  \
#                         - ( + 1/(dyV*(cv[i,3] - cv[S,3])))*psiNew[S]    

#     """STEP 3"""                     
#     l2_norm = 1.0
#     l2 = 1.0
#     l2_max = 1.0
#     it_number = 0
#     it_max = volnumb
#     rho = 1

#     r0 = r.copy()
#     u = r.copy()   #Direction vector
#     v = r.copy()   #Conjugate direction vector

#     while it_number < it_max and l2 > tol:

#         psiOld = psiNew.copy()
#         rk = r.copy()
#         uk = u.copy()
#         vk = v.copy()

#         """PASSO 4"""  # Calculating [A][D] (Au) from mazumder algorithm
#         Au = r.copy()
#         for i in range(volnumb):
#             dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
#             dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
#             W,N,E,S = idNeighbors.idNeighbors(volarr,i)
                
#             if W < 0:
#                 Au[i] = + (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                             - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*u[i] \
#                             + ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
#                             + ( + 1/(dyV*(cv[N,3] - cv[i,3])))*u[N]  \
#                             + ( + 1/(dyV*(cv[i,3] - cv[S,3])))*u[S]
#             elif E < 0:
#                 Au[i] = + (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                             - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*u[i] \
#                             + ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
#                             + ( + 1/(dyV*(cv[N,3] - cv[i,3])))*u[N]  \
#                             + ( + 1/(dyV*(cv[i,3] - cv[S,3])))*u[S]  
#             elif S < 0:
#                 Au[i] = + (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                             - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*u[i] \
#                             + ( + 1/(dxV*(cv[E,2] - cv[i,2])))*u[E]  \
#                             + ( + 1/(dxV*(cv[i,2] - cv[W,2])))*u[W]  \
#                             + ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*u[N]  
#             elif N < 0:
#                 Au[i] = + (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                             - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*u[i] \
#                             + ( + 1/(dxV*(cv[E,2] - cv[i,2])))*u[E]  \
#                             + ( + 1/(dxV*(cv[i,2] - cv[W,2])))*u[W]  \
#                             + ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*u[S]  
#             elif W < 0 and N < 0:
#                 Au[i] = + (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                             - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*u[i] \
#                             + ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
#                             + ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*u[S]  
#             elif W < 0 and S < 0:
#                 Au[i] = + (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                             - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*u[i] \
#                             + ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
#                             + ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*u[N]
#             elif E < 0 and N < 0:
#                 Au[i] = + (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                             - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*u[i] \
#                             + ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
#                             + ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*u[S]          
#             elif E < 0 and S < 0:
#                 Au[i] = + (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                             - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*u[i] \
#                             + ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
#                             + ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*u[N]  
#             else:
#                 Au[i] = + (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))\
#                             - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*u[i] \
#                             + (+ 1/(dxV*(cv[E,2] - cv[i,2])))*u[E]  \
#                             + (+ 1/(dxV*(cv[i,2] - cv[W,2])))*u[W]  \
#                             + (+ 1/(dyV*(cv[N,3] - cv[i,3])))*u[N]  \
#                             + (+ 1/(dyV*(cv[i,3] - cv[S,3])))*u[S]  

#         rho = np.sum(r0*rk)
#         rhok = np.sum(r0*Au)
#         alpha = rho/rhok
#         """PASSO 5"""
#         q = vk - alpha*Au
#         """PASSO 6"""
#         psiNew = psiOld + alpha*(vk + q) # Sobreescrevendo a iteração anterior
#         """PASSO 7"""
#         # Atualizando o resíduo 
        
#         for i in range(volnumb):
#             dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
#             dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
#             W,N,E,S = idNeighbors.idNeighbors(volarr,i)
#             if W < 0:
#                 r[i] = b[i] - (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                                 - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*psiNew[i] \
#                             - ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
#                             - ( + 1/(dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  \
#                             - ( + 1/(dyV*(cv[i,3] - cv[S,3])))*psiNew[S]
#             elif E < 0:
#                 r[i] = b[i] - (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                                 - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*psiNew[i] \
#                             - ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
#                             - ( + 1/(dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  \
#                             - ( + 1/(dyV*(cv[i,3] - cv[S,3])))*psiNew[S]  
#             elif S < 0:
#                 r[i] = b[i] - (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*psiNew[i] \
#                             - ( + 1/(dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
#                             - ( + 1/(dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
#                             - ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  
#             elif N < 0:
#                 r[i] = b[i] - (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*psiNew[i] \
#                             - ( + 1/(dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
#                             - ( + 1/(dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
#                             - ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*psiNew[S]  
#             elif W < 0 and N < 0:
#                 r[i] = b[i] - (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                                 - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*psiNew[i] \
#                             - ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
#                             - ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*psiNew[S]  
#             elif W < 0 and S < 0:
#                 r[i] = b[i] - (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                                 - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*psiNew[i] \
#                             - ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
#                             - ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*psiNew[N]
#             elif E < 0 and N < 0:
#                 r[i] = b[i] - (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                                 - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*psiNew[i] \
#                             - ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
#                             - ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*psiNew[S]          
#             elif E < 0 and S < 0:
#                 r[i] = b[i] - (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                                 - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*psiNew[i] \
#                             - ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
#                             - ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  
#             else:
#                 r[i] = b[i] - (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*psiNew[i] \
#                             - ( + 1/(dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
#                             - ( + 1/(dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
#                             - ( + 1/(dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  \
#                             - ( + 1/(dyV*(cv[i,3] - cv[S,3])))*psiNew[S]    
                        
#         l2_norm = np.sqrt(np.sum((r)**2))
#         l2_max = max(l2_norm,l2_max)
#         l2 = l2_norm/l2_max
#         """PASSO 8"""
#         rho = np.sum(r0*r)
#         beta = rho/np.sum(r0*rk)
#         """PASSO 9"""
#         v = r + beta*q
#         """PASSO 10"""
#         u = v + beta*(q + beta*uk)
        
#         # myfile.write('{} \t {:.5E} \n'.format(it_number,l2))

#         # print('{} \t {:.3E} '.format(it_number,l2))
#         # print(it_number,l2_norm)

#         it_number += 1

#     h_data[:volnumb,3] = psiNew

#     # myfile.close()

#     return h_data

@jit(nopython=True, parallel=True)
def poissonNew_cgs(volnumb,volarr,cv,h_data,tol=1e-6):

    """Pseudo-transient poisson equation"""

    """PASSO 1"""
    psiNew = h_data[:volnumb,3]#.astype('float128')#.copy()
    psiOld = psiNew.copy()
    b = np.zeros((volnumb))#.astype('float128')
    src = - h_data[:volnumb,4]#.astype('float128')#np.zeros((volnumb)).astype('float128')
    r = np.zeros((volnumb))#.astype('float128')
    q = np.zeros((volnumb))#.astype('float128')
    u = np.zeros((volnumb))#.astype('float128')
    v = np.zeros((volnumb))#.astype('float128')
    Au = np.zeros((volnumb))#.astype('float128')

    # myfile = open('./0 -Results_MazumderEx3-1/{}_convPhi_{}vols_tol{:.1E}.dat'.format(tag,volnumb,tol),'w')
    # myfile.write('Variables="it","R2"\n')
    
    """STEP 2"""
    # Construct b

    for i in range(volnumb):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        W,N,E,S = idNeighbors.idNeighbors(volarr,i)
        
        # bcLeft = 0.0
        # bcRight = 0.0#(1/4)*np.sinh(5) + (yH**2)*np.sinh(10*yH) + np.exp(2*cv[i,3])
        # bcBottom = 0.0#(1/4)*np.sinh(-5) + (xH**2)*np.sinh(10*xH) + 1
        # bcTop = 0.0#(1/4)*np.sinh(5) + (xH**2)*np.sinh(10*xH) + np.exp(2*cv[i,2])
        
        """ b """
        # if W < 0:
        #     b[i] = src[i] - 8*bcLeft/(3*dxV**2)
        # elif E < 0:
        #     b[i] = src[i] - 8*bcRight/(3*dxV**2)
        # elif S < 0:
        #     b[i] = src[i] - 8*bcBottom/(3*dyV**2)
        # elif N < 0:
        #     b[i] = src[i] - 8*bcTop/(3*dyV**2)
        # elif W < 0 and N < 0:
        #     b[i] = src[i] - (8/3)*(bcTop/dyV**2 + bcLeft/dxV**2)
        # elif W < 0 and S < 0:
        #     b[i] = src[i] - (8/3)*(bcLeft/dxV**2 + bcBottom/dyV**2)
        # elif E < 0 and N < 0:
        #     b[i] = src[i] - (8/3)*(bcRight/dxV**2 + bcTop/dyV**2)
        # elif E < 0 and S < 0:
        #     b[i] = src[i] - (8/3)*(bcRight/dxV**2 + bcBottom/dyV**2)
        # else:
        b[i] = src[i]
    
    # Inicializando o resíduo
    for i in range(volnumb):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        W,N,E,S = idNeighbors.idNeighbors(volarr,i)
        r[i] = b[i] - (+ (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*psiOld[i] \
                    - ( - 1/(dxV*(cv[E,2] - cv[i,2])))*psiOld[E]  \
                    - ( - 1/(dxV*(cv[i,2] - cv[W,2])))*psiOld[W]  \
                    - ( - 1/(dyV*(cv[N,3] - cv[i,3])))*psiOld[N]  \
                    - ( - 1/(dyV*(cv[i,3] - cv[S,3])))*psiOld[S]    

    """STEP 3"""                     
    l2_norm = 1.0
    l2 = 1.0
    l2_max = 1.0
    it_number = 0
    it_max = volnumb
    rho = 1

    r0 = r.copy()
    u = r.copy()   #Direction vector
    v = r.copy()   #Conjugate direction vector

    while it_number < it_max and l2 > tol:

        psiOld = psiNew.copy()
        rk = r.copy()
        uk = u.copy()
        vk = v.copy()

        """PASSO 4"""  # Calculating [A][D] (Au) from mazumder algorithm
        Au = r.copy()
        for i in range(volnumb):
            dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
            dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
            W,N,E,S = idNeighbors.idNeighbors(volarr,i)
                
            # if W < 0:
            #     Au[i] = + (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
            #                - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*u[i] \
            #                 + ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
            #                 + ( + 1/(dyV*(cv[N,3] - cv[i,3])))*u[N]  \
            #                 + ( + 1/(dyV*(cv[i,3] - cv[S,3])))*u[S]
            # elif E < 0:
            #     Au[i] = + (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
            #                - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*u[i] \
            #                 + ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
            #                 + ( + 1/(dyV*(cv[N,3] - cv[i,3])))*u[N]  \
            #                 + ( + 1/(dyV*(cv[i,3] - cv[S,3])))*u[S]  
            # elif S < 0:
            #     Au[i] = + (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
            #                - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*u[i] \
            #                 + ( + 1/(dxV*(cv[E,2] - cv[i,2])))*u[E]  \
            #                 + ( + 1/(dxV*(cv[i,2] - cv[W,2])))*u[W]  \
            #                 + ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*u[N]  
            # elif N < 0:
            #     Au[i] = + (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
            #                - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*u[i] \
            #                 + ( + 1/(dxV*(cv[E,2] - cv[i,2])))*u[E]  \
            #                 + ( + 1/(dxV*(cv[i,2] - cv[W,2])))*u[W]  \
            #                 + ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*u[S]  
            # elif W < 0 and N < 0:
            #     Au[i] = + (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
            #                - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*u[i] \
            #                 + ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
            #                 + ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*u[S]  
            # elif W < 0 and S < 0:
            #     Au[i] = + (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
            #                - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*u[i] \
            #                 + ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
            #                 + ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*u[N]
            # elif E < 0 and N < 0:
            #     Au[i] = + (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
            #                - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*u[i] \
            #                 + ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
            #                 + ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*u[S]          
            # elif E < 0 and S < 0:
            #     Au[i] = + (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
            #                - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*u[i] \
            #                 + ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
            #                 + ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*u[N]  
            # else:
            Au[i] = + (+ (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))\
                        + (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*u[i] \
                        + (- 1/(dxV*(cv[E,2] - cv[i,2])))*u[E]  \
                        + (- 1/(dxV*(cv[i,2] - cv[W,2])))*u[W]  \
                        + (- 1/(dyV*(cv[N,3] - cv[i,3])))*u[N]  \
                        + (- 1/(dyV*(cv[i,3] - cv[S,3])))*u[S]  

        rho = np.sum(r0*rk)
        rhok = np.sum(r0*Au)
        alpha = rho/rhok
        """PASSO 5"""
        q = vk - alpha*Au
        """PASSO 6"""
        psiNew = psiOld + alpha*(vk + q) # Sobreescrevendo a iteração anterior
        """PASSO 7"""
        # Atualizando o resíduo 
        
        for i in range(volnumb):
            dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
            dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
            W,N,E,S = idNeighbors.idNeighbors(volarr,i)
            # if W < 0:
            #     r[i] = b[i] - (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
            #                    - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*psiNew[i] \
            #                 - ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
            #                 - ( + 1/(dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  \
            #                 - ( + 1/(dyV*(cv[i,3] - cv[S,3])))*psiNew[S]
            # elif E < 0:
            #     r[i] = b[i] - (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
            #                    - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*psiNew[i] \
            #                 - ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
            #                 - ( + 1/(dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  \
            #                 - ( + 1/(dyV*(cv[i,3] - cv[S,3])))*psiNew[S]  
            # elif S < 0:
            #     r[i] = b[i] - (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
            #                    - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*psiNew[i] \
            #                 - ( + 1/(dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
            #                 - ( + 1/(dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
            #                 - ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  
            # elif N < 0:
            #     r[i] = b[i] - (- (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
            #                    - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*psiNew[i] \
            #                 - ( + 1/(dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
            #                 - ( + 1/(dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
            #                 - ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*psiNew[S]  
            # elif W < 0 and N < 0:
            #     r[i] = b[i] - (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
            #                    - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*psiNew[i] \
            #                 - ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
            #                 - ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*psiNew[S]  
            # elif W < 0 and S < 0:
            #     r[i] = b[i] - (- (4/(dxV))*(1/(cv[E,2] - cv[i,2]))     \
            #                    - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*psiNew[i] \
            #                 - ( + 4/(3*dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
            #                 - ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*psiNew[N]
            # elif E < 0 and N < 0:
            #     r[i] = b[i] - (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
            #                    - (4/(dyV))*(1/(cv[i,3] - cv[S,3])))*psiNew[i] \
            #                 - ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
            #                 - ( + 4/(3*dyV*(cv[i,3] - cv[S,3])))*psiNew[S]          
            # elif E < 0 and S < 0:
            #     r[i] = b[i] - (- (4/(dxV))*(1/(cv[i,2] - cv[W,2]))     \
            #                    - (4/(dyV))*(1/(cv[N,3] - cv[i,3])))*psiNew[i] \
            #                 - ( + 4/(3*dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
            #                 - ( + 4/(3*dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  
            # else:
            r[i] = b[i] - (+ (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*psiNew[i] \
                        - ( - 1/(dxV*(cv[E,2] - cv[i,2])))*psiNew[E]  \
                        - ( - 1/(dxV*(cv[i,2] - cv[W,2])))*psiNew[W]  \
                        - ( - 1/(dyV*(cv[N,3] - cv[i,3])))*psiNew[N]  \
                        - ( - 1/(dyV*(cv[i,3] - cv[S,3])))*psiNew[S]    
                        
        l2_norm = np.sqrt(np.sum((r)**2))
        l2_max = max(l2_norm,l2_max)
        l2 = l2_norm/l2_max
        """PASSO 8"""
        rho = np.sum(r0*r)
        beta = rho/np.sum(r0*rk)
        """PASSO 9"""
        v = r + beta*q
        """PASSO 10"""
        u = v + beta*(q + beta*uk)
        
        # myfile.write('{} \t {:.5E} \n'.format(it_number,l2))

        # print('{} \t {:.3E} '.format(it_number,l2))
        # print(it_number,l2_norm)

        it_number += 1

    h_data[:volnumb,3] = psiNew

    # myfile.close()

    return h_data