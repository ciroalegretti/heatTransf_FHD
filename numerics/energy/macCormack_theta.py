#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:40:53 2021

@author: alegretti
"""

import numpy as np
from numba import jit,prange
from pre import idNeighbors
from numerics.energy import fluxTheta

@jit(nopython=True, parallel=True)
def fluxTheta_C(flux,h_data,t_data,wf,cv,volArr):
    
    k = len(flux)
    
    for i in prange(k):
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)
        
        # Flux in E volume direction
        flux[i,0] = + (wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])*(wf[i,1]*t_data[i,1] + (1 - wf[i,1])*t_data[E,1]) 
        # Flux in N volume direction
        flux[i,1] = + (wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])*(wf[i,2]*t_data[i,1] + (1 - wf[i,2])*t_data[N,1]) 
        # Flux in W volume direction
        flux[i,2] = + (wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])*(wf[i,3]*t_data[i,1] + (1 - wf[i,3])*t_data[W,1]) 
        # Flux in S volume direction
        flux[i,3] = + (wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])*(wf[i,4]*t_data[i,1] + (1 - wf[i,4])*t_data[S,1])
        
    return flux


@jit(nopython=True, parallel=True)
def fluxTheta_D(flux,t_data,c,wf,cv,volArr):

    k = len(flux)
    
    for i in prange(k):
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)

        # Flux in E volume direction
        flux[i,0] = + (1./c)*(t_data[E,1] - t_data[i,1])/(cv[E,2] - cv[i,2])
        # Flux in N volume direction
        flux[i,1] = + (1./c)*(t_data[N,1] - t_data[i,1])/(cv[N,3] - cv[i,3])
        # Flux in W volume direction
        flux[i,2] = + (1./c)*(t_data[i,1] - t_data[W,1])/(cv[i,2] - cv[W,2])
        # Flux in S volume direction
        flux[i,3] = + (1./c)*(t_data[i,1] - t_data[S,1])/(cv[i,3] - cv[S,3])

    return flux

# @jit(nopython=True, parallel=True)
def macCormack_theta(m_data,h_data,t_data,c,wf,volArr,cv,dt,phi,Ec,Re,Pe):
    """ Solves vorticity equation using explicit MacCormack method"""

    n = len(volArr)
    flux = np.zeros((len(cv),4))

    t_dataP = t_data.copy()
    t_dataC = t_data.copy()
    
    derP = np.zeros((n,1))
    derC = np.zeros((n,1))
    derAVG = np.zeros((n,1))
    
    # Predictor step (forward approximation for advection fluxes)
    
    fluxconvP = fluxTheta_C(flux,h_data,t_data,wf,cv,volArr)
    fluxdiffP = fluxTheta_D(flux,t_data,c,wf,cv,volArr)
    
    for i in prange(n):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        
        derP[i,0] = - fluxconvP[i,0]/dxV \
                    - fluxconvP[i,1]/dyV \
                    + (fluxdiffP[i,0] - fluxdiffP[i,2])/dxV \
                    + (fluxdiffP[i,1] - fluxdiffP[i,3])/dyV    

        t_dataP[i,1] = t_data[i,1] + derP[i,0]*dt

    # Corrector step (backward approximation for advection fluxes)

    fluxconvC = fluxTheta_C(flux,h_data,t_dataP,wf,cv,volArr)
    fluxdiffC = fluxTheta_D(flux,t_dataP,c,wf,cv,volArr)

    for i in prange(n):
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]

        derC[i,0] = + fluxconvC[i,2]/dxV \
                    + fluxconvC[i,3]/dyV \
                    + (fluxdiffC[i,0] - fluxdiffC[i,2])/dxV \
                    + (fluxdiffC[i,1] - fluxdiffC[i,3])/dyV  

        # Computing average time derivative 
        derAVG[i,0] = (derP[i,0] + derC[i,0])/2# + (9/4)*t_data[i,3]*phi*(Ec/(Re*Pe))*( \
                                                  # (m_data[i,5]*m_data[i,4] - m_data[i,6]*m_data[i,3])*(h_data[E,2] - h_data[W,2])/(cv[E,2] - cv[W,2]) + \
                                                  # (m_data[i,5]*m_data[i,4] - m_data[i,6]*m_data[i,3])*(h_data[N,1] - h_data[S,1])/(cv[N,3] - cv[S,3]) \
                                                  #     )
        
        # Final step 
        t_dataC[i,1] = t_data[i,1] + derAVG[i,0]*dt

    return t_dataC