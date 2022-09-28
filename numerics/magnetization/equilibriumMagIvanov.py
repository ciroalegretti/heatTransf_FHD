#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:43:50 2021

@author: alegretti
"""
from numba import jit
import numpy as np

@jit(nopython=True)
def equilibriumMagIvanov(mData,tData,phi,alpha0,beta,volNumb,lambda0):
    """
    magData = [#vol, M0x, M0y, Hx, Hy, Mx, My]
               
    """
    Md = 1.0                # Solid magnetization of the suspended particles
    
    # External Field applied only after sudden expansion        
    for i in range(volNumb):
        
        """ Calculating modified alpha_langevin """
        # tData[i,-1] = alpha0/(beta*tData[i,1] + 1.)
        tData[i,-2] = alpha0*(np.sqrt(mData[i,3]**2 + mData[i,4]**2))/(beta*tData[i,1] + 1.)
        tData[i,2] = lambda0#/(beta*tData[i,1] + 1.)
        
        """ Tese Rafael """
        alpha = tData[i,-2]
        L = 1.0/(np.tanh(alpha)) - 1.0/(alpha)
        g = L*( - alpha*(1/np.sinh(alpha))**2 + 1/alpha)
        h = L**2*(2*alpha**2*(1/np.tanh(alpha))*(1/np.sinh(alpha)**2) - 2/alpha)
        z = g**2/L
        
        M0 = (Md*L)*phi +\
             ((4*np.pi/3)*Md*g)*tData[i,2]*phi**2 +\
             ((8*(np.pi**2)/9)*h + (16*(np.pi**2)/144)*z )*(tData[i,2]**2)*Md*phi**3
        
        M0x = M0*mData[i,3]/(np.sqrt(mData[i,3]**2 + mData[i,4]**2))     # Equilibrium Magnetization - x direction
        M0y = M0*mData[i,4]/(np.sqrt(mData[i,3]**2 + mData[i,4]**2))     # Equilibrium Magnetization - y direction
        mData[i,1] = M0x
        mData[i,2] = M0y

    return mData,tData