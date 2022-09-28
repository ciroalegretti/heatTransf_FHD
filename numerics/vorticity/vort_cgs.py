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
def vort_cgs(volnumb,wf,volarr,cv,h_data,m_data,t_data,dt,re,pe,phi,tol=1e-3):
    
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
        
        b[i] = Wo[i]/dt + (9*t_data[i,-2]*phi/(2*re*pe))*( \
                                                # flux of kelvin forces right frontier
                                              + ( - (wf[i,1]*m_data[i,5] + (1 - wf[i,1])*m_data[E,5])*(m_data[E,3] - m_data[i,3])/(cv[E,2] - cv[i,2]) \
                                                  + (wf[i,1]*m_data[i,6] + (1 - wf[i,1])*m_data[E,6])*(m_data[N,3] + m_data[NE,3] - m_data[S,3] - m_data[SE,3])/(4*dyV) \
                                                              
                                                # flux of kelvin forces left frontier              
                                                  - (wf[i,3]*m_data[i,5] + (1 - wf[i,3])*m_data[W,5])*(m_data[i,3] - m_data[W,3])/(cv[i,2] - cv[W,2]) \
                                                  + (wf[i,3]*m_data[i,6] + (1 - wf[i,3])*m_data[W,6])*(m_data[S,3] + m_data[SW,3] - m_data[N,3] - m_data[NW,3])/(4*dyV))/dxV \
                                                    
                                                # flux of kelvin forces top frontier
                                              + ( - (wf[i,2]*m_data[i,5] + (1 - wf[i,2])*m_data[N,5])*(m_data[W,4] + m_data[NW,4] - m_data[E,4] - m_data[NE,4])/(4*dxV) \
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


@jit(nopython=True, parallel=True)
def vort_cgs_LDC(volnumb,wf,volarr,cv,h_data,m_data,t_data,dt,re,pe,phi,tol=1e-3):
    
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
        
        b[i] = Wo[i]/dt + (9*t_data[i,-2]*phi/(2*re*pe))*( \
                                                # Kelvin force by Generalized Gauss Theorem (integrating along volume faces)
                                                (1/(2*dxV**2))*((m_data[E,5] + m_data[i,5])*(m_data[E,4] - m_data[i,4]) - (m_data[i,5] + m_data[W,5])*(m_data[i,4] - m_data[W,4]))
                                              + (1/(2*dyV**2))*((m_data[N,6] + m_data[i,6])*(m_data[N,3] - m_data[i,3]) - (m_data[i,6] + m_data[S,6])*(m_data[i,3] - m_data[S,3]))
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