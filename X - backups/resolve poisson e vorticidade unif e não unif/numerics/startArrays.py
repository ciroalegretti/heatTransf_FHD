#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:26:16 2021

@author: alegretti
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def starters1d(volArr,CVdata):
    """ Return 2D array data:
        
       data    = [#vol, u, v, psi, w]
       magData = [#vol, M0x, M0y, Hx, Hy, Mx, My]
       thermoData = [#vol, theta, chi, alphaLang]
       
       flux_[] = [E,N,W,S]   -> fluxes through each face
       s_[] = [s_i]          -> source terms
        
    """
    n = len(volArr) # internal volumes
    lastGhost = np.min(volArr) 
    gi = np.abs(lastGhost) # last ghost abs index
    nt = n + gi    # total internal and ghost volumes
    
    data = np.zeros((nt,5))
    magData = np.zeros((nt,7))
    thermoData = np.zeros((nt,4))
    
    
    # fluxes arrays
    flux_xi = np.zeros((n,4))
    flux_mx = np.zeros((n,4))
    flux_my = np.zeros((n,4))
    flux_t = np.zeros((n,4))
    flux_psi = np.zeros((n,4))
    # source-terms arrays
    st_xi = np.zeros((n,1))
    st_mx = np.zeros((n,1))
    st_my = np.zeros((n,1))
    st_t = np.zeros((n,1))
    st_psi = np.zeros((n,1))
    
    for i in range(nt):
        data[i,0] = CVdata[i,0]
        magData[i,0] = CVdata[i,0]
        thermoData[i,0] = CVdata[i,0]
    
    return data,magData,thermoData,flux_xi,flux_mx,flux_my,flux_t,flux_psi,st_xi,st_mx,st_my,st_t,st_psi
