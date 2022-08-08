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
            data[W,4] = 6.*(2.*controlVol[i,3] - (STEP + H))/(h**2)
    
    "step wall"
    for i in range(edgeVol,len(volArr)):
        W = volArr[i,1]
        if W < 0:
            E = volArr[i,3]
            data[W,4] = 2.*(data[W,3] - data[E,3])/((controlVol[i,4])**2) - data[i,4]
            
    """Top BC"""
    for i in range(len(volArr)):
        N = volArr[i,2] 
        if N < 0:  
            S = volArr[i,4] 
            data[N,4] = 2.*(data[N,3] - data[S,3])/((controlVol[i,5])**2) - data[i,4]
            
    """Outflow, Neumann BC for completelly developed flow"""
    for i in range(len(volArr)):
        E = volArr[i,3]
        if E < 0:
            data[E,4] = data[i,4]
            
    """Bottom walls BC"""
    for i in range(len(volArr)):
        S = volArr[i,4] 
        if S < 0:    
            N = volArr[i,2]
            data[S,4] = 2.*(data[S,3] - data[N,3])/((controlVol[i,5])**2) - data[i,4]
            
    
    return data


@jit(nopython=True)
def bc_w_PP(data,volArr,controlVol,h):
    
    """Inlet BC"""
    for i in range(len(volArr)):
        W = volArr[i,1]
        if W < 0:
            data[W,4] = 6.*(2.*controlVol[i,3] - h)/(h**2)
            # data[W,4] = data[i,4]
    
    """Top BC"""
    for i in range(len(volArr)):
        N = volArr[i,2] 
        if N < 0:  
            S = volArr[i,4] 
            data[N,4] = 2.*(data[N,3] - data[S,3])/((controlVol[i,5])**2) - data[i,4]
            
    """Outflow, Neumann BC for completelly developed flow"""
    for i in range(len(volArr)):
        E = volArr[i,3]
        if E < 0:
            data[E,4] = data[i,4]
            
    """Bottom walls BC"""
    for i in range(len(volArr)):
        S = volArr[i,4] 
        if S < 0:    
            N = volArr[i,2]
            data[S,4] = 2.*(data[S,3] - data[N,3])/((controlVol[i,5])**2) - data[i,4]
    
    return data

@jit(nopython=True)
def bc_w_LDC(data,volArr,controlVol,h,U):

    """Lid"""
    for i in range(len(volArr)):
        N = volArr[i,2] 
        if N < 0:  
            S = volArr[i,4] 
            data[N,4] = 2.*(data[N,3] - data[S,3])/((controlVol[i,5])**2) - data[i,4] - U*4/controlVol[i,5]       # Thom's formula

    """Left wall"""
    for i in range(len(volArr)):
        W = volArr[i,1]
        if W < 0:
            E = volArr[i,3]
            data[W,4] = 2.*(data[W,3] - data[E,3])/((controlVol[i,5])**2) - data[i,4]       # Thom's formula

    """Right wall"""
    for i in range(len(volArr)):
        E = volArr[i,3]
        if E < 0:
            W = volArr[i,1]
            data[E,4] = 2.*(data[E,3] - data[W,3])/((controlVol[i,5])**2) - data[i,4]       # Thom's formula

    """ Bottom wall"""
    for i in range(len(volArr)):
        S = volArr[i,4] 
        if S < 0:    
            N = volArr[i,2]
            data[S,4] = 2.*(data[S,3] - data[N,3])/((controlVol[i,5])**2) - data[i,4]       # Thom's formula

    return data
    
@jit(nopython=True)
def bc_w_LDC_new(data,volArr,controlVol,h):
###################################################
#
#	Thom formula 
#	(1sr order Taylor Series expansion to extrapolate ghosts from boundary):
#
#	d**2 (psi)/dn^2 = - xi
#
####################################################


    """Lid"""
    for i in range(len(volArr)):
        N = volArr[i,2] 
        if N < 0:  
            S = volArr[i,4] 
            data[N,4] = 2.*(data[N,3] - data[S,3])/((controlVol[i,5])**2) - data[i,4] - 4/controlVol[i,5]

    """Left wall"""
    for i in range(len(volArr)):
        W = volArr[i,1]
        if W < 0:
            E = volArr[i,3]
            data[W,4] = 2.*(data[W,3] - data[E,3])/((controlVol[i,5])**2) - data[i,4]

    """Right wall"""
    for i in range(len(volArr)):
        E = volArr[i,3]
        if E < 0:
            W = volArr[i,1]
            data[E,4] = 2.*(data[E,3] - data[W,3])/((controlVol[i,5])**2) - data[i,4]

    """ Bottom wall"""
    for i in range(len(volArr)):
        S = volArr[i,4] 
        if S < 0:    
            N = volArr[i,2]
            data[S,4] = 2.*(data[S,3] - data[N,3])/((controlVol[i,5])**2) - data[i,4]

    return data
