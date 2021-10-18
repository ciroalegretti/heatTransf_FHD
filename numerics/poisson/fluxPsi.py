#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 03:22:33 2021

@author: alegretti
"""
from numba import jit,prange
from pre import idNeighbors

@jit(nopython=True)
def fluxdiffPsi(flux,h_data,cv,volArr):

    n = len(volArr)
    
    for i in prange(n):
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)
        
        # Flux in E volume direction
        flux[i,0] = + (h_data[E,3] - h_data[i,3])/(cv[E,2] - cv[i,2])
        # Flux in N volume direction
        flux[i,1] = + (h_data[N,3] - h_data[i,3])/(cv[N,3] - cv[i,3])
        # Flux in W volume direction
        flux[i,2] = + (h_data[i,3] - h_data[W,3])/(cv[i,2] - cv[W,2])
        # Flux in S volume direction
        flux[i,3] = + (h_data[i,3] - h_data[S,3])/(cv[i,3] - cv[S,3])
        
    return flux
