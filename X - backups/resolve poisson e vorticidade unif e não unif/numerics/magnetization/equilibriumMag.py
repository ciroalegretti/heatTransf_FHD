#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:43:50 2021

@author: alegretti
"""
from numba import jit
import numpy as np

@jit(nopython=True)
def equilibriumMag(magData,phi,Md,alpha,Hx,Hy,nx1,ny1,volNumb,cv,Le):
    """ Calculating equilibrium magnetization"""
    M0 = phi*Md*(1.0/(np.tanh(alpha)) - 1.0/(alpha))
    M0x = M0*Hx/(np.sqrt(Hx**2 + Hy**2))     # Equilibrium Magnetization - x direction
    M0y = M0*Hy/(np.sqrt(Hx**2 + Hy**2))     # Equilibrium Magnetization - y direction
    
    # External Field applied only after sudden expansion        
    for i in range(volNumb):
        if cv[i,2]>Le:
            magData[i,1] = M0x
            magData[i,2] = M0y
            magData[i,3] = Hx
            magData[i,4] = Hy
    
    # External Field applied to the whole domain  
    # magData[:,1] = M0x
    # magData[:,2] = M0y
    # magData[:,3] = Hx
    # magData[:,4] = Hy
    
  
    return magData