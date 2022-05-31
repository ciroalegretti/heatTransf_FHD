#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:44:56 2021

@author: alegretti
"""

import numpy as np
from numba import jit,prange
from numerics.vorticity import fluxVort
from pre import idNeighbors

@jit(nopython=True, parallel=True)
def fluxHydro_C(flux,h_data,wf,cv,volArr):
    
    flux_c = flux.copy()

    k = len(flux)
    
    for i in prange(k):
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)
        
        # Flux in E volume direction
        flux_c[i,0] = + (wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])*(wf[i,1]*h_data[i,4] + (1 - wf[i,1])*h_data[E,4]) 
        # Flux in N volume direction
        flux_c[i,1] = + (wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])*(wf[i,2]*h_data[i,4] + (1 - wf[i,2])*h_data[N,4]) 
        # Flux in W volume direction
        flux_c[i,2] = + (wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])*(wf[i,3]*h_data[i,4] + (1 - wf[i,3])*h_data[W,4]) 
        # Flux in S volume direction
        flux_c[i,3] = + (wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])*(wf[i,4]*h_data[i,4] + (1 - wf[i,4])*h_data[S,4])
        
    return flux_c


@jit(nopython=True, parallel=True)
def fluxHydro_D(flux,h_data,c,wf,cv,volArr):

    k = len(flux)
    
    for i in prange(k):
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)

        # Flux in E volume direction
        flux[i,0] = + (1./c)*(h_data[E,4] - h_data[i,4])/(cv[E,2] - cv[i,2])
        # Flux in N volume direction
        flux[i,1] = + (1./c)*(h_data[N,4] - h_data[i,4])/(cv[N,3] - cv[i,3])
        # Flux in W volume direction
        flux[i,2] = + (1./c)*(h_data[i,4] - h_data[W,4])/(cv[i,2] - cv[W,2])
        # Flux in S volume direction
        flux[i,3] = + (1./c)*(h_data[i,4] - h_data[S,4])/(cv[i,3] - cv[S,3])

    return flux

@jit(nopython=True, parallel=True)
def macCormack_w(data,magData,re,wf,volArr,cv,dt,pe,phi):
    """ Solves vorticity equation using explicit MacCormack method"""

    n = len(volArr)
    flux = np.zeros((n,4))

    dataP = data.copy()
    dataC = data.copy()
    
    derP = np.zeros((n,1))
    derC = np.zeros((n,1))
    derAVG = np.zeros((n,1))
    
    # Predictor step (forward approximation for advection fluxes)
    
    fluxconvP = fluxHydro_C(flux,data,wf,cv,volArr)
    fluxdiffP = fluxHydro_D(flux,data,re,wf,cv,volArr)
    
    for i in prange(n):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        
        derP[i,0] = - fluxconvP[i,0]/dxV \
                    - fluxconvP[i,1]/dyV \
                    + (fluxdiffP[i,0] - fluxdiffP[i,2])/dxV \
                    + (fluxdiffP[i,1] - fluxdiffP[i,3])/dyV    

        dataP[i,4] = data[i,4] + derP[i,0]*dt

    # Corrector step (backward approximation for advection fluxes)

    fluxconvC = fluxHydro_C(fluxconvP,dataP,wf,cv,volArr)
    fluxdiffC = fluxHydro_D(fluxdiffP,dataP,re,wf,cv,volArr)

    for i in prange(n):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]

        derC[i,0] = + fluxconvC[i,2]/dxV \
                    + fluxconvC[i,3]/dyV \
                    + (fluxdiffC[i,0] - fluxdiffC[i,2])/dxV \
                    + (fluxdiffC[i,1] - fluxdiffC[i,3])/dyV  
        
        # Computing average time derivative 
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)
        NW,NE,SE,SW = idNeighbors.idDiagonalVol(volArr,i)

        derAVG[i,0] = (derP[i,0] + derC[i,0])/2 + (9*magData[i,-1]*phi/(2*re*pe))*( \
                                                # flux of kelvin forces right frontier
                                                + ( (wf[i,1]*magData[i,5] + (1 - wf[i,1])*magData[E,5])*(magData[E,3] - magData[i,3])/(cv[E,2] - cv[i,2]) \
                                                        + (wf[i,1]*magData[i,6] + (1 - wf[i,1])*magData[E,6])*(magData[N,3] + magData[NE,3] - magData[S,3] - magData[SE,3])/(4*dyV) \
                                                              
                                                # flux of kelvin forces left frontier              
                                                  - (wf[i,3]*magData[i,5] + (1 - wf[i,3])*magData[W,5])*(magData[i,3] - magData[W,3])/(cv[i,2] - cv[W,2]) \
                                                        + (wf[i,3]*magData[i,6] + (1 - wf[i,3])*magData[W,6])*(magData[S,3] + magData[SW,3] - magData[N,3] - magData[NW,3])/(4*dyV))/dxV \
                                                    
                                                # flux of kelvin forces top frontier
                                                - ( (wf[i,2]*magData[i,5] + (1 - wf[i,2])*magData[N,5])*(magData[W,4] + magData[NW,4] - magData[E,4] - magData[NE,4])/(4*dxV) \
                                                        + (wf[i,2]*magData[i,6] + (1 - wf[i,2])*magData[E,6])*(magData[N,4] - magData[i,4])/(cv[N,3] - cv[i,3]) \
                                                              
                                                # flux of kelvin forces bottom frontier              
                                                  - (wf[i,4]*magData[i,5] + (1 - wf[i,4])*magData[N,5])*(magData[E,4] + magData[SE,4] - magData[W,4] - magData[SW,4])/(4*dxV) \
                                                        + (wf[i,4]*magData[i,6] + (1 - wf[i,4])*magData[E,6])*(magData[i,4] - magData[S,4])/(cv[i,3] - cv[S,3]))/dyV \
                                                            
                                                # Fluxes of -div(MxH)
                                                - ((magData[E,5]*magData[E,4] - magData[i,5]*magData[i,4])/(cv[E,2] - cv[i,2]) -  \
                                                   (magData[i,5]*magData[i,4] - magData[W,5]*magData[W,4])/(cv[i,2] - cv[W,2]))/dxV/2 \
                                                - ((magData[N,5]*magData[N,4] - magData[i,5]*magData[i,4])/(cv[N,3] - cv[i,3]) - \
                                                   (magData[i,5]*magData[i,4] - magData[S,5]*magData[S,4])/(cv[i,3] - cv[S,3]))/dyV/2)
        
        # Final step 
        dataC[i,4] = data[i,4] + derAVG[i,0]*dt

    return dataC