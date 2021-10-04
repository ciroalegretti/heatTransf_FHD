#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:57:13 2021

@author: alegretti
"""
from numba import jit

@jit(nopython=True)
def bc_theta(tData,volArr,controlVol,edgeVol,STEP):    
    
    h = 0.5
    
    """Inlet BC"""
    for i in range(edgeVol):
        W = volArr[i,1]
        if W < 0:
            # tData[W,1] = - tData[i,1]
            tData[W,1] = - controlVol[i,3]/h + 1 + STEP/h    # Linear temperature velocity profile ranging from 0 (top wall) to 1 (bottom wall)
            
    """Step face"""
    for i in range(edgeVol,len(volArr)):
        W = volArr[i,1]
        if W < 0:
            # tData[W,1] = - tData[i,1] # enforcing theta = 0.0
            tData[W,1] = 2. - tData[i,1] # enforcing theta = 1.0
            
    """Top BC"""
    for i in range(len(volArr)):
        N = volArr[i,2] 
        if N < 0:    
            tData[N,1] = - tData[i,1] # enforcing theta = 0.0
            # tData[N,1] = 2. - tData[i,1] # enforcing theta = 1.0
            
    """Outflow, Neumann BC for completelly developed theta"""
    for i in range(len(volArr)):
        E = volArr[i,3]
        if E < 0:
            tData[E,1] = tData[i,1]
            
    """ Bottom walls"""
    #"""upstream theta BC"""
    for i in range(edgeVol):
        S = volArr[i,4] 
        if S < 0:    
            # tData[S,1] = - tData[i,1] # enforcing theta = 0.0
            tData[S,1] = 2. - tData[i,1] # enforcing theta = 1.0
            
    """downstream theta BC"""
    for i in range(edgeVol,len(volArr)):
        S = volArr[i,4] 
        if S < 0:    
            # tData[S,1] = - tData[i,1] # enforcing theta = 0.0
            tData[S,1] = 2. - tData[i,1] # enforcing theta = 1.0
 
    return tData