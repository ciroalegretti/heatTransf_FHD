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
def magX_cgs(volnumb,wf,volarr,cv,h_data,m_data,t_data,dt,pe,phi,tol=1e-3):
    
    #        magData = [#vol, M0x, M0y, Hx, Hy, Mx, My]
    
    MX_o = m_data[:,-2].copy()
    m_dataP = m_data.copy()
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
        # NW,NE,SE,SW = idNeighbors.idDiagonalVol(volarr,i)
        
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        
        b[i] = m_dataP[i,-2]/dt - h_data[i,4]*m_data[i,6]/2. \
                                 + (m_data[i,3]*m_data[i,6]**2 - (m_data[i,5]*m_data[i,6]*m_data[i,4]))*3*t_data[i,-1]/(4*pe) \
                                 + (m_data[i,1] - m_data[i,5])/pe
                                                

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
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])   )*m_dataP[i,-2] \
                    - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) )*m_dataP[E,-2]  \
                    - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) )*m_dataP[W,-2]  \
                    - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) )*m_dataP[N,-2]  \
                    - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) )*m_dataP[S,-2]                        
                            
    r0 = r.copy()
    rho = 1#np.sum(r0*r)
    
    while it_number < it_max and l2_norm > tol:
        
        Wk = m_dataP[:,-2].copy()
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
                                - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])   )*p[i] \
                        + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) )*p[E]  \
                        + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) )*p[W]  \
                        + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) )*p[N]  \
                        + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) )*p[S]                                  
        
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
                                - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])   )*u[i] \
                        + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) )*u[E]  \
                        + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) )*u[W]  \
                        + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) )*u[N]  \
                        + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) )*u[S]                                     
           
            Aq[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                                - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                                + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                                - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])   )*q[i] \
                        + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) )*q[E]  \
                        + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) )*q[W]  \
                        + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) )*q[N]  \
                        + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) )*q[S]                                  
           
        
        r = rk - alpha*(Au + Aq)
        MX_o = Wk + alpha*(u + q)
        
        l2_norm = np.sqrt(np.sum((r)**2))
        
        it_number += 1
        
    m_data[:,-2] = MX_o
    
    return m_data

@jit(nopython=True, parallel=True)
def magY_cgs(volnumb,wf,volarr,cv,h_data,m_data,t_data,dt,pe,phi,tol=1e-3):
    
    #        magData = [#vol, M0x, M0y, Hx, Hy, Mx, My]
    
    MY_o = m_data[:,-1].copy()
    m_dataP = m_data.copy()
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
        # NW,NE,SE,SW = idNeighbors.idDiagonalVol(volarr,i)
        
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        
        b[i] = m_dataP[i,-1]/dt + h_data[i,4]*m_dataP[i,5]/2. \
                                + (m_data[i,4]*(m_data[i,5]**2) - m_data[i,5]*m_data[i,6]*m_data[i,4] )*3*t_data[i,-1]/(4*pe)   \
                                + (m_data[i,2] - m_data[i,6])/pe
                                                

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
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])   )*m_dataP[i,-1] \
                    - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) )*m_dataP[E,-1]  \
                    - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) )*m_dataP[W,-1]  \
                    - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) )*m_dataP[N,-1]  \
                    - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) )*m_dataP[S,-1]                        
                            
    r0 = r.copy()
    rho = 1#np.sum(r0*r)
    
    while it_number < it_max and l2_norm > tol:
        
        Wk = m_dataP[:,-1].copy()
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
                                - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])   )*p[i] \
                        + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) )*p[E]  \
                        + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) )*p[W]  \
                        + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) )*p[N]  \
                        + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) )*p[S]                                  
        
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
                                - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])   )*u[i] \
                        + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) )*u[E]  \
                        + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) )*u[W]  \
                        + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) )*u[N]  \
                        + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) )*u[S]                                     
           
            Aq[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                                - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                                + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                                - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])   )*q[i] \
                        + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) )*q[E]  \
                        + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) )*q[W]  \
                        + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) )*q[N]  \
                        + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) )*q[S]                                  
           
        
        r = rk - alpha*(Au + Aq)
        MY_o = Wk + alpha*(u + q)
        
        l2_norm = np.sqrt(np.sum((r)**2))
        
        it_number += 1
        
    m_data[:,-1] = MY_o
    
    return m_data