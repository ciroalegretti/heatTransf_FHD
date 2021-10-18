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
def poisson_cg(volnumb,volarr,cv,h_data,tol=1e-6):
    
    # ny, nx = psi.shape
    
    r = np.zeros((len(cv)))
    d = np.zeros((len(cv)))
    q = np.zeros((len(cv)))

    b = - h_data[:,4]

    l2_norm = 1.0
    
    it_number = 0
    it_max = volnumb
    
    for i in prange(volnumb):
        W,N,E,S = idNeighbors.idNeighbors(volarr,i)
        
        r[i] = b[i] - ((- 2/(cv[E,2] - cv[i,2]) - 2/(cv[i,2] - cv[W,2]))/(cv[E,2] - cv[W,2]) + ( - 2/(cv[N,3] - cv[i,3]) - 2/(cv[i,3] - cv[S,3]))/(cv[N,3] - cv[S,3]))*h_data[i,3] \
                       - (2/((cv[E,2] - cv[W,2])*(cv[E,2] - cv[i,2])))*h_data[E,3] \
                       - (2/((cv[E,2] - cv[W,2])*(cv[i,2] - cv[W,2])))*h_data[W,3] \
                       - (2/((cv[N,3] - cv[S,3])*(cv[N,3] - cv[i,3])))*h_data[N,3] \
                       - (2/((cv[N,3] - cv[S,3])*(cv[i,3] - cv[S,3])))*h_data[S,3]                             
    d = r.copy()
    delta_new = np.sum(r*r)
    delta_old = delta_new
    
    
    while it_number < it_max and l2_norm > tol:
        
        psik = h_data[:,3]
        rk = r.copy()
        dk = d.copy()

        for i in prange(volnumb):
            W,N,E,S = idNeighbors.idNeighbors(volarr,i)
            
            q[i] = ((- 2/(cv[E,2] - cv[i,2]) - 2/(cv[i,2] - cv[W,2]))/(cv[E,2] - cv[W,2]) + ( - 2/(cv[N,3] - cv[i,3]) - 2/(cv[i,3] - cv[S,3]))/(cv[N,3] - cv[S,3]))*d[i] \
                       + (2/((cv[E,2] - cv[W,2])*(cv[E,2] - cv[i,2])))*d[E] \
                       + (2/((cv[E,2] - cv[W,2])*(cv[i,2] - cv[W,2])))*d[W] \
                       + (2/((cv[N,3] - cv[S,3])*(cv[N,3] - cv[i,3])))*d[N] \
                       + (2/((cv[N,3] - cv[S,3])*(cv[i,3] - cv[S,3])))*d[S]   
        
        alpha = delta_new/np.sum(d*q)
        psi = psik + alpha*dk
        r = rk - alpha*q
        delta_old = delta_new
        delta_new = np.sum(r*r)
        beta = delta_new/delta_old
        d = r + beta*dk
        
        l2_norm = np.sqrt(np.sum((r)**2))
        
        it_number += 1
        
        
    h_data[:,3] = psi
        
    return h_data