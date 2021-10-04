#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:44:56 2021

@author: alegretti
"""

import numpy as np
from numba import jit
from numerics.poisson import fluxPsi
from numerics.vorticity import fluxVort
from pre import idNeighbors

# @jit(nopython=True)
# def poissonExplicit(data,volArr,volNodes,CV):

#     dataPrev = data.copy()
    
#     for i in range(len(volArr)):
#         W,N,E,S = idNeighbors.idNeighbors(volArr,i)

#         a = (CV[E,2] - CV[W,2])/(CV[N,3] - CV[i,3])
#         b = (CV[E,2] - CV[W,2])/(CV[i,3] - CV[S,3])
#         c = (CV[N,3] - CV[S,3])/(CV[E,2] - CV[i,2])
#         d = (CV[N,3] - CV[S,3])/(CV[i,2] - CV[W,2])
        
#         data[i,3] = (1./(a + b + c + d))*(2.*dataPrev[i,4]*CV[i,1] + a*dataPrev[N,3] + b*dataPrev[S,3] + c*dataPrev[E,3] + d*dataPrev[W,3])
            
#     return data

@jit(nopython=True)
def macCormack_w(data,c,flux,s,wf,volArr,cv,dt):
    """ Solves vorticity equation using explicit MacCormack method"""

    n = len(volArr)

    dataP = data.copy()
    dataC = data.copy()
    
    derP = np.zeros((n,1))
    derC = np.zeros((n,1))
    derAVG = np.zeros((n,1))
    
    # Predictor step (forward approximation for advection fluxes)
    
    fluxconvP = fluxVort.fluxHydro_C(flux,data,wf,cv,volArr)
    fluxdiffP = fluxVort.fluxHydro_D(flux,data,c,wf,cv,volArr)
    
    for i in range(n):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        
        derP[i,0] = - fluxconvP[i,0]/dxV \
                    - fluxconvP[i,1]/dyV \
                    + (fluxdiffP[i,0] - fluxdiffP[i,2])/dxV \
                    + (fluxdiffP[i,1] - fluxdiffP[i,3])/dyV    

        dataP[i,4] = data[i,4] + derP[i,0]*dt

    # Corrector step (backward approximation for advection fluxes)

    fluxconvC = fluxVort.fluxHydro_C(fluxconvP,dataP,wf,cv,volArr)
    fluxdiffC = fluxVort.fluxHydro_D(fluxdiffP,dataP,c,wf,cv,volArr)

    for i in range(n):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]

        derC[i,0] = - fluxconvC[i,2]/dxV \
                    - fluxconvC[i,3]/dyV \
                    + (fluxdiffC[i,0] - fluxdiffC[i,2])/dxV \
                    + (fluxdiffC[i,1] - fluxdiffC[i,3])/dyV  

        # Computing average time derivative 
        derAVG[i,0] = (derP[i,0] + derC[i,0])/2
        
        # Final step 
        dataC[i,4] = data[i,4] + derAVG[i,0]*dt

    return dataC