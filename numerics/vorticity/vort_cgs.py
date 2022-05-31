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
def vort_cgs(volnumb,wf,volarr,cv,h_data,m_data,dt,re,pe,phi,tol=1e-6):
    
    Wo = h_data[:,4].copy()
    b = np.zeros((len(cv)))
    r = np.zeros((len(cv)))
    p = np.zeros((len(cv)))
    q = np.zeros((len(cv)))
    u = np.zeros((len(cv)))
    v = np.zeros((len(cv)))
    Au = np.zeros((len(cv)))
    Aq = np.zeros((len(cv)))
    
    for i in prange(volnumb):
        W,N,E,S = idNeighbors.idNeighbors(volarr,i)
        NW,NE,SE,SW = idNeighbors.idDiagonalVol(volarr,i)
        
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        
        b[i] = Wo[i]/dt + (9*m_data[i,-1]*phi/(2*re*pe))*( \
                                                # flux of kelvin forces right frontier
                                                + ( (wf[i,1]*m_data[i,5] + (1 - wf[i,1])*m_data[E,5])*(m_data[E,3] - m_data[i,3])/(cv[E,2] - cv[i,2]) \
                                                + (wf[i,1]*m_data[i,6] + (1 - wf[i,1])*m_data[E,6])*(m_data[N,3] + m_data[NE,3] - m_data[S,3] - m_data[SE,3])/(4*dyV) \
                                                              
                                                # flux of kelvin forces left frontier              
                                                  - (wf[i,3]*m_data[i,5] + (1 - wf[i,3])*m_data[W,5])*(m_data[i,3] - m_data[W,3])/(cv[i,2] - cv[W,2]) \
                                                  + (wf[i,3]*m_data[i,6] + (1 - wf[i,3])*m_data[W,6])*(m_data[S,3] + m_data[SW,3] - m_data[N,3] - m_data[NW,3])/(4*dyV))/dxV \
                                                    
                                                # flux of kelvin forces top frontier
                                                - ( (wf[i,2]*m_data[i,5] + (1 - wf[i,2])*m_data[N,5])*(m_data[W,4] + m_data[NW,4] - m_data[E,4] - m_data[NE,4])/(4*dxV) \
                                                + (wf[i,2]*m_data[i,6] + (1 - wf[i,2])*m_data[E,6])*(m_data[N,4] - m_data[i,4])/(cv[N,3] - cv[i,3]) \
                                                              
                                                # flux of kelvin forces bottom frontier              
                                                  - (wf[i,4]*m_data[i,5] + (1 - wf[i,4])*m_data[N,5])*(m_data[E,4] + m_data[SE,4] - m_data[W,4] - m_data[SW,4])/(4*dxV) \
                                                  + (wf[i,4]*m_data[i,6] + (1 - wf[i,4])*m_data[E,6])*(m_data[i,4] - m_data[S,4])/(cv[i,3] - cv[S,3]))/dyV \
                                                            
                                                # Fluxes of -div(MxH)
                                                - ((m_data[E,5]*m_data[E,4] - m_data[i,5]*m_data[i,4])/(cv[E,2] - cv[i,2]) -  \
                                                   (m_data[i,5]*m_data[i,4] - m_data[W,5]*m_data[W,4])/(cv[i,2] - cv[W,2]))/dxV/2 \
                                                - ((m_data[N,5]*m_data[N,4] - m_data[i,5]*m_data[i,4])/(cv[N,3] - cv[i,3]) - \
                                                   (m_data[i,5]*m_data[i,4] - m_data[S,5]*m_data[S,4])/(cv[i,3] - cv[S,3]))/dyV/2)

    l2_norm = 1.0
    
    it_number = 0
    it_max = len(cv)
    
    for i in prange(volnumb):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        W,N,E,S = idNeighbors.idNeighbors(volarr,i)
        
        r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                            - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                            + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
                            + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*Wo[i] \
                    - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
                    - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
                    - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*dyV*(cv[N,3] - cv[i,3])))*Wo[N]  \
                    - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*dyV*(cv[i,3] - cv[S,3])))*Wo[S]                        
                            
    r0 = r.copy()
    rho = 1#np.sum(r0*r)
    
    while it_number < it_max and l2_norm > tol:
        
        Wk = Wo.copy()
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
            
            v[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                           - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                           + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                           - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
                           + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                           + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*p[i] \
                    + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*dxV*(cv[E,2] - cv[i,2])))*p[E]  \
                    + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*dxV*(cv[i,2] - cv[W,2])))*p[W]  \
                    + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*dyV*(cv[N,3] - cv[i,3])))*p[N]  \
                    + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*dyV*(cv[i,3] - cv[S,3])))*p[S]                                
        
        sigma = np.sum(r0*v)
        alpha = rho/sigma
        q = u - alpha*v
        
        for i in prange(volnumb):
            dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
            dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
            W,N,E,S = idNeighbors.idNeighbors(volarr,i)
            
            Au[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                            - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                            + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
                            + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*u[i] \
                    + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
                    + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
                    + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*dyV*(cv[N,3] - cv[i,3])))*u[N]  \
                    + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*dyV*(cv[i,3] - cv[S,3])))*u[S]                                
           
            Aq[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                            - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                            + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
                            + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*q[i] \
                    + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*dxV*(cv[E,2] - cv[i,2])))*q[E]  \
                    + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*dxV*(cv[i,2] - cv[W,2])))*q[W]  \
                    + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*dyV*(cv[N,3] - cv[i,3])))*q[N]  \
                    + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*dyV*(cv[i,3] - cv[S,3])))*q[S]                                
           
        
        r = rk - alpha*(Au + Aq)
        Wo = Wk + alpha*(u + q)
        
        l2_norm = np.sqrt(np.sum((r)**2))
        
        it_number += 1
        
    h_data[:,4] = Wo
    
    return h_data

# @jit(nopython=True, parallel=True)
# def w_cgs(volnumb,volarr,cv,h_data,tol,re,dt,wf):

#     """Fully implicit poisson equation"""

#     """PASSO 1"""
#     Wo = h_data[:volnumb,4]#.astype('float128')#.copy()
#     b = np.zeros((volnumb))#.astype('float128')
#     src = np.zeros((volnumb))#.astype('float128')
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
        
#         src[i] = Wo[i]/dt
        
#         bcLeft = (8/dxV**2)*(0 - h_data[i,3] + dxV*0)
#         bcRight = (8/dxV**2)*(0 - h_data[i,3] + dxV*0)
#         bcBottom = (8/dyV**2)*(0 - h_data[i,3] + dyV*0)
#         bcTop = (8/dyV**2)*(0 - h_data[i,3] - dyV*1/2)
        
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
#             r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                 - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                 + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                 - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                 + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                 + (4/(re*dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                                 + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*Wo[i] \
#                         - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  - 4/(3*re**dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
#                         - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  - 1/(re*dyV*(cv[N,3] - cv[i,3])))*Wo[N]  \
#                         - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  - 1/(re*dyV*(cv[i,3] - cv[S,3])))*Wo[S]
#         elif E < 0:
#             r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                 - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                 + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                 - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                 + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                 + (4/(re*dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                                 + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*Wo[i] \
#                         - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 4/(re*3*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
#                         - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  - 1/(re*dyV*(cv[N,3] - cv[i,3])))*Wo[N]  \
#                         - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  - 1/(re*dyV*(cv[i,3] - cv[S,3])))*Wo[S]  
#         elif S < 0:
#             r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                 - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                 + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                 - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                 + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                 + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 + (4/(re*dyV))*(1/(cv[N,3] - cv[i,3])))*Wo[i] \
#                         - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  - 1/(re*dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
#                         - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  - 1/(re*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
#                         - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 4/(re*3*dyV*(cv[N,3] - cv[i,3])))*Wo[N]  
#         elif N < 0:
#             r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                 - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                 + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                 - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                 + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                 + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                + (4/(re*dyV))*(1/(cv[i,3] - cv[S,3])))*Wo[i] \
#                         - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  - 1/(re*dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
#                         - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  - 1/(re*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
#                         - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  - 4/(re*3*dyV*(cv[i,3] - cv[S,3])))*Wo[S]  
#         elif W < 0 and N < 0:
#             r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                 - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                 + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                 - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                 + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                 + (4/(re*dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                            + (4/(re*dyV))*(1/(cv[i,3] - cv[S,3])))*Wo[i] \
#                         - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  - 4/(re*3*dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
#                         - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  - 4/(re*3*dyV*(cv[i,3] - cv[S,3])))*Wo[S]  
#         elif W < 0 and S < 0:
#             r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                 - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                 + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                 - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                 + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                 + (4/(re*dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                            + (4/(re*dyV))*(1/(cv[N,3] - cv[i,3])))*Wo[i] \
#                         - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  - 4/(re*3*dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
#                         - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  - 4/(re*3*dyV*(cv[N,3] - cv[i,3])))*Wo[N]
#         elif E < 0 and N < 0:
#             r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                 - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                 + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                 - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                 + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                 + (4/(re*dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                            + (4/(re*dyV))*(1/(cv[i,3] - cv[S,3])))*Wo[i] \
#                         - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  - 4/(re*3*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
#                         - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  - 4/(re*3*dyV*(cv[i,3] - cv[S,3])))*Wo[S]          
#         elif E < 0 and S < 0:
#             r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                 - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                 + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                 - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                 + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                 + (4/(re*dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                                 + (4/(re*dyV))*(1/(cv[N,3] - cv[i,3])))*Wo[i] \
#                         - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  - 4/(re*3*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
#                         - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  - 4/(re*3*dyV*(cv[N,3] - cv[i,3])))*Wo[N]  
#         else:
#             r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                 - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                 + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                 - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                 + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*Wo[i] \
#                         - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
#                         - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
#                         - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*dyV*(cv[N,3] - cv[i,3])))*Wo[N]  \
#                         - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*dyV*(cv[i,3] - cv[S,3])))*Wo[S]

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

#         wOld = Wo.copy()
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
#                 Au[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (4/(re*dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                                + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*u[i] \
#                         + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 4/(3*re**dxV*(cv[E,2] - cv[i,2])))*u[E]  \
#                         + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*dyV*(cv[N,3] - cv[i,3])))*u[N]  \
#                         + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*dyV*(cv[i,3] - cv[S,3])))*u[S]
#             elif E < 0:
#                 Au[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (4/(re*dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                                + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*u[i] \
#                         + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 4/(re*3*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
#                         + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*dyV*(cv[N,3] - cv[i,3])))*u[N]  \
#                         + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*dyV*(cv[i,3] - cv[S,3])))*u[S]  
#             elif S < 0:
#                 Au[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                + (4/(re*dyV))*(1/(cv[N,3] - cv[i,3])))*u[i] \
#                         + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
#                         + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
#                         + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 4/(re*3*dyV*(cv[N,3] - cv[i,3])))*u[N]  
#             elif N < 0:
#                 Au[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (4/(re*dyV))*(1/(cv[i,3] - cv[S,3])))*u[i] \
#                         + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
#                         + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
#                         + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 4/(re*3*dyV*(cv[i,3] - cv[S,3])))*u[S]   
#             elif W < 0 and N < 0:
#                 Au[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (4/(re*dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                                     + (4/(re*dyV))*(1/(cv[i,3] - cv[S,3])))*u[i] \
#                         + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 4/(re*3*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
#                         + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 4/(re*3*dyV*(cv[i,3] - cv[S,3])))*u[S]
#             elif W < 0 and S < 0:
#                 Au[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (4/(re*dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                                     + (4/(re*dyV))*(1/(cv[N,3] - cv[i,3])))*u[i] \
#                         + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 4/(re*3*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
#                         + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 4/(re*3*dyV*(cv[N,3] - cv[i,3])))*u[N]
#             elif E < 0 and N < 0:
#                 Au[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (4/(re*dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                                     + (4/(re*dyV))*(1/(cv[i,3] - cv[S,3])))*u[i] \
#                          + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 4/(re*3*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
#                          + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 4/(re*3*dyV*(cv[i,3] - cv[S,3])))*u[S]           
#             elif E < 0 and S < 0:
#                 Au[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (4/(re*dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                                     + (4/(re*dyV))*(1/(cv[N,3] - cv[i,3])))*u[i] \
#                          + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 4/(re*3*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
#                          + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 4/(re*3*dyV*(cv[N,3] - cv[i,3])))*u[N]  
#             else:
#                 Au[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                 - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                 + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                 - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                 + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                 + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*u[i] \
#                         + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
#                         + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
#                         + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*dyV*(cv[N,3] - cv[i,3])))*u[N]  \
#                         + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*dyV*(cv[i,3] - cv[S,3])))*u[S]

#         rho = np.sum(r0*rk)
#         rhok = np.sum(r0*Au)
#         alpha = rho/rhok
#         """PASSO 5"""
#         q = vk - alpha*Au
#         """PASSO 6"""
#         u = wOld + alpha*(vk + q) # Sobreescrevendo a iteração anterior
#         """PASSO 7"""
#         # Atualizando o resíduo 
        
#         for i in range(volnumb):
#             dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
#             dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
#             W,N,E,S = idNeighbors.idNeighbors(volarr,i)
#             if W < 0:
#                 r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (4/(re*dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*Wo[i] \
#                             - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  - 4/(3*re**dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
#                             - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  - 1/(re*dyV*(cv[N,3] - cv[i,3])))*Wo[N]  \
#                             - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  - 1/(re*dyV*(cv[i,3] - cv[S,3])))*Wo[S]
#             elif E < 0:
#                 r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (4/(re*dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*Wo[i] \
#                             - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 4/(re*3*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
#                             - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  - 1/(re*dyV*(cv[N,3] - cv[i,3])))*Wo[N]  \
#                             - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  - 1/(re*dyV*(cv[i,3] - cv[S,3])))*Wo[S]  
#             elif S < 0:
#                 r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (4/(re*dyV))*(1/(cv[N,3] - cv[i,3])))*Wo[i] \
#                             - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  - 1/(re*dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
#                             - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  - 1/(re*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
#                             - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 4/(re*3*dyV*(cv[N,3] - cv[i,3])))*Wo[N]  
#             elif N < 0:
#                 r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                    + (4/(re*dyV))*(1/(cv[i,3] - cv[S,3])))*Wo[i] \
#                             - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  - 1/(re*dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
#                             - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  - 1/(re*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
#                             - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  - 4/(re*3*dyV*(cv[i,3] - cv[S,3])))*Wo[S]  
#             elif W < 0 and N < 0:
#                 r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (4/(re*dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                                + (4/(re*dyV))*(1/(cv[i,3] - cv[S,3])))*Wo[i] \
#                             - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  - 4/(re*3*dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
#                             - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  - 4/(re*3*dyV*(cv[i,3] - cv[S,3])))*Wo[S]  
#             elif W < 0 and S < 0:
#                 r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (4/(re*dxV))*(1/(cv[E,2] - cv[i,2]))     \
#                                + (4/(re*dyV))*(1/(cv[N,3] - cv[i,3])))*Wo[i] \
#                             - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  - 4/(re*3*dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
#                             - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  - 4/(re*3*dyV*(cv[N,3] - cv[i,3])))*Wo[N]
#             elif E < 0 and N < 0:
#                 r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (4/(re*dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                                + (4/(re*dyV))*(1/(cv[i,3] - cv[S,3])))*Wo[i] \
#                             - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  - 4/(re*3*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
#                             - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  - 4/(re*3*dyV*(cv[i,3] - cv[S,3])))*Wo[S]          
#             elif E < 0 and S < 0:
#                 r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3]))
#                                     + (4/(re*dxV))*(1/(cv[i,2] - cv[W,2]))     \
#                                     + (4/(re*dyV))*(1/(cv[N,3] - cv[i,3])))*Wo[i] \
#                             - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  - 4/(re*3*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
#                             - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  - 4/(re*3*dyV*(cv[N,3] - cv[i,3])))*Wo[N]  
#             else:
#                 r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
#                                     - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
#                                     + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
#                                     - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
#                                     + (1/(re*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
#                                     + (1/(re*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*Wo[i] \
#                             - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*dxV*(cv[E,2] - cv[i,2])))*Wo[E]  \
#                             - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*dxV*(cv[i,2] - cv[W,2])))*Wo[W]  \
#                             - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*dyV*(cv[N,3] - cv[i,3])))*Wo[N]  \
#                             - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*dyV*(cv[i,3] - cv[S,3])))*Wo[S]
                        
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
#         print(it_number,l2_norm)

#         it_number += 1

#     h_data[:volnumb,4] = Wo

#     # myfile.close()

#     return h_data