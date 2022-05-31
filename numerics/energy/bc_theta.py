#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:57:13 2021

@author: alegretti
"""
from numba import jit

@jit(nopython=True)
def bc_thetaBFS(tData,volArr,controlVol,edgeVol,STEP):    
    
    h = 0.5
    
    """Inlet BC"""
    for i in range(edgeVol):
        W = volArr[i,1]
        if W < 0:
            tData[W,1] = 2 - tData[i,1] # Enforcing theta = 1
            # tData[W,1] = 0.0#- tData[i,1]
            # tData[W,1] = - controlVol[i,3]/h + 1 + STEP/h    # Linear temperature velocity profile ranging from 0 (top wall) to 1 (bottom wall)
            
    """Step face"""
    for i in range(edgeVol,len(volArr)):
        W = volArr[i,1]
        if W < 0:
            # tData[W,1] = tData[i,1] # enforcing dTheta/dn = 0 adiabatic
            # tData[W,1] = 2. - tData[i,1] # enforcing theta = 1.0
            tData[W,1] = - tData[i,1] # enforcing theta = 0.0
            
    """Top BC"""
    for i in range(len(volArr)):
        N = volArr[i,2] 
        if N < 0:    
            tData[N,1] = - tData[i,1] # enforcing theta = 0.0
            # tData[N,1] = tData[i,1] # enforcing dTheta/dn = 0 adiabatic
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
            # tData[S,1] = tData[i,1] # enforcing dTheta/dn = 0 adiabatic
            # tData[S,1] = 2. - tData[i,1] # enforcing theta = 1.0
            tData[S,1] = - tData[i,1] # enforcing theta = 0.0
            
    #"""downstream theta BC"""
    for i in range(edgeVol,len(volArr)):
        S = volArr[i,4] 
        if S < 0:    
            tData[S,1] = - tData[i,1] # enforcing theta = 0.0
            # tData[S,1] = 2. - tData[i,1] # enforcing theta = 1.0
 
    return tData


@jit(nopython=True)
def bc_theta_PP(tData,volArr,controlVol):    
    
    """Inlet BC"""
    for i in range(len(volArr)):
        W = volArr[i,1]
        if W < 0:
            # tData[W,1] = - tData[i,1] # Enforcing theta = 0
            tData[W,1] = 2 - tData[i,1] # Enforcing theta = 1
            # tData[W,1] = - controlVol[i,3]/h + 1 + STEP/h    # Linear temperature velocity profile ranging from 0 (top wall) to 1 (bottom wall)
            
    """Top BC"""
    for i in range(len(volArr)):
        N = volArr[i,2] 
        if N < 0:    
            tData[N,1] = - tData[i,1] # enforcing theta = 0.0
            # tData[N,1] = tData[i,1] # enforcing dTheta/dn = 0 adiabatic
            # tData[N,1] = 2. - tData[i,1] # enforcing theta = 1.0
            
    """Outflow, Neumann BC for completelly developed theta"""
    for i in range(len(volArr)):
        E = volArr[i,3]
        if E < 0:
            tData[E,1] = tData[i,1]
            
    """ Bottom walls"""
    for i in range(len(volArr)):
        S = volArr[i,4] 
        if S < 0:    
            tData[S,1] = - tData[i,1] # enforcing theta = 0.0
            # tData[S,1] = tData[i,1] # enforcing dTheta/dn = 0 adiabatic
            # tData[S,1] = 2. - tData[i,1] # enforcing theta = 1.0
            
    return tData

@jit(nopython=True)
def bc_theta_LDC(tData,volArr,controlVol):    
    
    """Left Wall"""
    for i in range(len(volArr)):
        W = volArr[i,1]
        if W < 0:
            # tData[W,1] = - tData[i,1] # enforcing theta = 0.0
            # tData[W,1] = - controlVol[i,3]/h + 1 + STEP/h    # Linear temperature velocity profile ranging from 0 (top wall) to 1 (bottom wall)
            tData[W,1] = tData[i,1] # enforcing dTheta/dn = 0 adiabatic
            
    """Top Lid"""
    for i in range(len(volArr)):
        N = volArr[i,2] 
        if N < 0:    
            tData[N,1] = - tData[i,1] # enforcing theta = 0.0
            # tData[N,1] = tData[i,1] # enforcing dTheta/dn = 0 adiabatic
            # tData[N,1] = 2. - tData[i,1] # enforcing theta = 1.0
            
    """Right wall"""
    for i in range(len(volArr)):
        E = volArr[i,3]
        if E < 0:
#             tData[E,1] = - tData[i,1] # enforcing theta = 0.0
            tData[W,1] = tData[i,1] # enforcing dTheta/dn = 0 adiabatic
            
    """ Bottom wall"""
    for i in range(len(volArr)):
        S = volArr[i,4] 
        if S < 0:    
            # tData[S,1] = tData[i,1] # enforcing dTheta/dn = 0 adiabatic
            tData[S,1] = 2. - tData[i,1] # enforcing theta = 1.0
            
    return tData
