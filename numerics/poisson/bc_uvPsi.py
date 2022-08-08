#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:53:34 2021

@author: alegretti
"""
from numba import jit

@jit(nopython=True)
def bc_uvPsi(data,volArr,controlVol,edgeVol,STEP):    

    h = 0.5
    H = STEP + h

    """Haggen-Poiseuille Inlet BC"""
    for i in range(edgeVol):
        W = volArr[i,1]
        if W < 0:
            data[W,1] = - 6.*(controlVol[i,3]**2 - (STEP + H)*controlVol[i,3] + STEP*H)/(h**2)
            data[W,2] = 0.0
            data[W,3] = - 6.*((controlVol[i,3]**3 - STEP**3)/3 - (STEP + H)*(controlVol[i,3]**2 - STEP**2)/2 + STEP*H*(controlVol[i,3] - STEP))/(h**2)
            
    """Step face"""
    for i in range(edgeVol,len(volArr)):
        W = volArr[i,1]
        if W < 0:
            data[W,1] = - data[i,1]
            data[W,2] = - data[i,2]
            data[W,3] = - data[i,3]
            
    """Top BC"""
    for i in range(len(volArr)):
        N = volArr[i,2] 
        if N < 0:    
            data[N,1] = - data[i,1]
            data[N,2] = - data[i,2]
            data[N,3] = 1.0 - data[i,3]
            
    """Outflow, Neumann BC for completelly developed flow"""
    for i in range(len(volArr)):
        E = volArr[i,3]
        if E < 0:
            data[E,1] = data[i,1]
            data[E,2] = data[i,2]
            data[E,3] = data[i,3]
            
    """ Bottom walls"""
    # FLOW BC
    for i in range(len(volArr)):
        S = volArr[i,4] 
        if S < 0:    
            data[S,1] = - data[i,1]
            data[S,2] = - data[i,2]
            data[S,3] = - data[i,3]
 
    return data

@jit(nopython=True)
def HP_uvPsi_PP(data,volArr,controlVol,h):    
    
    """ Set initial fields as Hagen-Poiseuille analytical solution"""
    
    for i in range(len(volArr)):
        data[i,1] = - 6.*(controlVol[i,3]**2 - h*controlVol[i,3])/(h**2)
        data[i,2] = 0
        data[i,3] = - 6.*((controlVol[i,3]**3)/3 - h*(controlVol[i,3]**2)/2)/(h**2)
        # data[i,4] = 6.*(2.*controlVol[i,3] - h)/(h**2)

    return data

@jit(nopython=True)
def bc_uvPsi_PP(data,volArr,controlVol,h):    

    """Haggen-Poiseuille Inlet BC"""
    for i in range(len(volArr)):
        W = volArr[i,1]
        if W < 0:
            data[W,1] = - 6.*(controlVol[i,3]**2 - h*controlVol[i,3])/(h**2)
            data[W,2] = 0.0
            data[W,3] = - 6.*((controlVol[i,3]**3)/3 - h*(controlVol[i,3]**2)/2)/(h**2)
            
            # data[W,1] = data[i,1]
            # data[W,2] = data[i,2]
            # data[W,3] = data[i,3]
            
    """Top BC"""
    for i in range(len(volArr)):
        N = volArr[i,2] 
        if N < 0:    
            data[N,1] = - data[i,1]
            data[N,2] = - data[i,2]
            data[N,3] = 2.0*h - data[i,3] #psi_top = 2*(U*h) - psi_i
            
    """Outflow, Neumann BC for completelly developed flow"""
    for i in range(len(volArr)):
        E = volArr[i,3]
        if E < 0:
            data[E,1] = data[i,1]
            data[E,2] = data[i,2]
            data[E,3] = data[i,3]
            
    """ Bottom walls"""
    # FLOW BC
    for i in range(len(volArr)):
        S = volArr[i,4] 
        if S < 0:    
            data[S,1] = - data[i,1]
            data[S,2] = - data[i,2]
            data[S,3] = - data[i,3]
 
    return data

@jit(nopython=True)
def bc_uvPsi_LDC(data,volArr,controlVol,h,U):    

    """Lid"""
    for i in range(len(volArr)):
        N = volArr[i,2] 
        if N < 0:    
            data[N,1] = + U*2.0 - data[i,1]
            data[N,2] = - data[i,2]
            data[N,3] = - data[i,3]

    """Left wall"""
    for i in range(len(volArr)):
        W = volArr[i,1]
        if W < 0:
            data[W,1] = - data[i,1]
            data[W,2] = - data[i,2]
            data[W,3] = - data[i,3]
            
    """Right wall"""
    for i in range(len(volArr)):
        E = volArr[i,3]
        if E < 0:
            data[E,1] = - data[i,1]
            data[E,2] = - data[i,2]
            data[E,3] = - data[i,3]
            
    """ Bottom wall"""
    for i in range(len(volArr)):
        S = volArr[i,4] 
        if S < 0:    
            data[S,1] = - data[i,1]
            data[S,2] = - data[i,2]
            data[S,3] = - data[i,3]
 
    return data
