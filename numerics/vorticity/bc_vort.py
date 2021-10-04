#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:55:41 2021

@author: alegretti
"""
from numba import jit

@jit(nopython=True)
def bc_w(data,volArr,controlVol,edgeVol,STEP):
    
    h = 0.5
    H = STEP + h
    
    """Inlet BC"""
    for i in range(edgeVol):
        W = volArr[i,1]
        if W < 0:
            data[W,4] = 6.*(2.*controlVol[i,3] - (STEP + H))/(h**3)
    
    "step wall"
    for i in range(edgeVol,len(volArr)):
        W = volArr[i,1]
        if W < 0:
            data[W,4] = 8.*(data[W,3] - data[i,3])/((controlVol[W,2] - controlVol[i,2])**2) - data[i,4]
            
    """Top BC"""
    for i in range(len(volArr)):
        N = volArr[i,2] 
        if N < 0:  
            data[N,4] = 8.*(data[N,3] - data[i,3])/((controlVol[N,3] - controlVol[i,3])**2) - data[i,4]
            
    """Outflow, Neumann BC for completelly developed flow"""
    for i in range(len(volArr)):
        E = volArr[i,3]
        if E < 0:
            data[E,4] = data[i,4]
            
    """Bottom walls BC"""
    for i in range(len(volArr)):
        S = volArr[i,4] 
        if S < 0:    
            data[S,4] = 8.*(data[S,3] - data[i,3])/((controlVol[S,3] - controlVol[i,3])**2) - data[i,4]
            
    
    return data