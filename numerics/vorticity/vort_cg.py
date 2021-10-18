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
def vort_cg(volnumb,volarr,cv,h_data,dt,re,tol=1e-6):
    
    Wo = h_data[:,4].copy()
    Wn = Wo.copy()
    b = np.zeros((len(cv)))
    r = np.zeros((len(cv)))
    d = np.zeros((len(cv)))
    q = np.zeros((len(cv)))
    
    for i in prange(volnumb):
        b[i] = Wn[i]/dt

    l2_norm = 1.0
    
    it_number = 0
    it_max = len(cv)
    
    for i in prange(volnumb):
        W,N,E,S = idNeighbors.idNeighbors(volarr,i)
        
        Upos = np.maximum(h_data[i,1],0)
        Uneg = np.minimum(h_data[i,1],0)
        Vpos = np.maximum(h_data[i,2],0)
        Vneg = np.minimum(h_data[i,2],0)
        wFoX = (Wo[E] - Wo[i])/(cv[E,2] - cv[i,2])
        wBaX = (Wo[i] - Wo[W])/(cv[i,2] - cv[W,2])
        wFoY = (Wo[N] - Wo[i])/(cv[N,3] - cv[i,3])
        wBaY = (Wo[i] - Wo[S])/(cv[i,3] - cv[S,3])
        
        r[i] = b[i] - (1./dt + (2./re)*(1./((cv[E,2] - cv[W,2])*(cv[E,2] - cv[i,2])) + 1./((cv[E,2] - cv[W,2])*(cv[i,2] - cv[W,2])) + 1./((cv[N,3] - cv[S,3])*(cv[N,3] - cv[i,3])) + 1./((cv[N,3] - cv[S,3])*(cv[i,3] - cv[S,3]))))*Wo[i] \
                    - (- 2./(re*(cv[E,2] - cv[W,2])*(cv[E,2] - cv[i,2])))*Wo[E]  \
                    - (- 2./(re*(cv[E,2] - cv[W,2])*(cv[i,2] - cv[W,2])))*Wo[W]  \
                    - (- 2./(re*(cv[N,3] - cv[S,3])*(cv[N,3] - cv[i,3])))*Wo[N]  \
                    - (- 2./(re*(cv[N,3] - cv[S,3])*(cv[i,3] - cv[S,3])))*Wo[S]  \
                    - (Upos*wBaX + Uneg*wFoX)                                 \
                    - (Vpos*wBaY + Vneg*wFoY)                                  
                            
    d = r.copy()
    delta_new = np.sum(r*r)
    delta_old = delta_new
    
    
    while it_number < it_max and l2_norm > tol:
        
        Wk = Wo.copy()
        rk = r.copy()
        dk = d.copy()

        for i in prange(volnumb):
            W,N,E,S = idNeighbors.idNeighbors(volarr,i)
            
            Upos = np.maximum(h_data[i,1],0)
            Uneg = np.minimum(h_data[i,1],0)
            Vpos = np.maximum(h_data[i,2],0)
            Vneg = np.minimum(h_data[i,2],0)
            dFoX = (d[E] - d[i])/(cv[E,2] - cv[i,2])
            dBaX = (d[i] - d[W])/(cv[i,2] - cv[W,2])
            dFoY = (d[N] - d[i])/(cv[N,3] - cv[i,3])
            dBaY = (d[i] - d[S])/(cv[i,3] - cv[S,3])
            
            q[i] =  + (1./dt + (2./re)*(1./((cv[E,2] - cv[W,2])*(cv[E,2] - cv[i,2])) + 1./((cv[E,2] - cv[W,2])*(cv[i,2] - cv[W,2])) + 1./((cv[N,3] - cv[S,3])*(cv[N,3] - cv[i,3])) + 1./((cv[N,3] - cv[S,3])*(cv[i,3] - cv[S,3]))))*d[i] \
                    + (- 2./(re*(cv[E,2] - cv[W,2])*(cv[E,2] - cv[i,2])))*d[E]  \
                    + (- 2./(re*(cv[E,2] - cv[W,2])*(cv[i,2] - cv[W,2])))*d[W]  \
                    + (- 2./(re*(cv[N,3] - cv[S,3])*(cv[N,3] - cv[i,3])))*d[N]  \
                    + (- 2./(re*(cv[N,3] - cv[S,3])*(cv[i,3] - cv[S,3])))*d[S]  \
                    + (Upos*dBaX + Uneg*dFoX)                                 \
                    + (Vpos*dBaY + Vneg*dFoY)                          
    
        alpha = delta_new/np.sum(d*q)
        Wo = Wk + alpha*dk
        r = rk - alpha*q
        delta_old = delta_new
        delta_new = np.sum(r*r)
        beta = delta_new/delta_old
        d = r + beta*dk
        
        l2_norm = np.sqrt(np.sum((r)**2))
        
        it_number += 1
    
    h_data[:,4] = Wo
        
    return h_data