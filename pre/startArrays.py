#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:26:16 2021

@author: alegretti
"""
import numpy as np
from numba import jit
from pre import clegg


@jit(nopython=True)
def starters1d(volArr,CVdata,Hx,Hy,alpha0):
    """ Return 2D array data:
        
       data    = [#vol, u, v, psi, w]
       magData = [#vol, M0x, M0y, Hx, Hy, Mx, My]
       thermoData = [#vol, theta, chi, alphaLang]
       
       flux = [E,N,W,S]   -> fluxes through each face
        
    """
    n = len(volArr) # internal volumes
    lastGhost = np.min(volArr) 
    gi = np.abs(lastGhost) # last ghost abs index
    nt = n + gi    # total internal and ghost volumes
    
    data = np.zeros((nt,5))
    magData = np.zeros((nt,7))
    thermoData = np.zeros((nt,4))
    
    # fluxes arrays
    flux = np.zeros((n,4))
    
    for i in range(n):
        data[i,0] = CVdata[i,0]
        magData[i,0] = CVdata[i,0]
        magData[i,3] = Hx
        magData[i,4] = Hy
        thermoData[i,0] = CVdata[i,0]
        thermoData[i,-1] = alpha0
    
    return data,magData,thermoData,flux
