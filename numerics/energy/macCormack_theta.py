#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:40:53 2021

@author: alegretti
"""

import numpy as np
from numba import jit
from pre import idNeighbors
from numerics.energy import fluxTheta

@jit(nopython=True)
def macCormack_theta(m_data,h_data,t_data,c,flux,wf,volArr,cv,dt,phi,Ec,Re,Pe):
    """ Solves vorticity equation using explicit MacCormack method"""

    n = len(volArr)

    t_dataP = t_data.copy()
    t_dataC = t_data.copy()
    
    derP = np.zeros((n,1))
    derC = np.zeros((n,1))
    derAVG = np.zeros((n,1))
    
    # Predictor step (forward approximation for advection fluxes)
    
    fluxconvP = fluxTheta.fluxTheta_C(flux,h_data,t_data,wf,cv,volArr)
    fluxdiffP = fluxTheta.fluxTheta_D(flux,t_data,c,wf,cv,volArr)
    
    for i in range(n):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        
        derP[i,0] = - fluxconvP[i,0]/dxV \
                    - fluxconvP[i,1]/dyV \
                    + (fluxdiffP[i,0] - fluxdiffP[i,2])/dxV \
                    + (fluxdiffP[i,1] - fluxdiffP[i,3])/dyV    

        t_dataP[i,1] = t_data[i,1] + derP[i,0]*dt

    # Corrector step (backward approximation for advection fluxes)

    fluxconvC = fluxTheta.fluxTheta_C(flux,h_data,t_dataP,wf,cv,volArr)
    fluxdiffC = fluxTheta.fluxTheta_D(flux,t_dataP,c,wf,cv,volArr)

    for i in range(n):
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]

        derC[i,0] = + fluxconvC[i,2]/dxV \
                    + fluxconvC[i,3]/dyV \
                    + (fluxdiffC[i,0] - fluxdiffC[i,2])/dxV \
                    + (fluxdiffC[i,1] - fluxdiffC[i,3])/dyV  

        # Computing average time derivative 
        derAVG[i,0] = (derP[i,0] + derC[i,0])/2 + (9/2)*m_data[i,3]*phi*(Ec/(Re*Pe))*( \
                                                  (m_data[i,5]*m_data[i,4] - m_data[i,4]*m_data[i,5])*(h_data[E,2] - h_data[W,2])/(cv[E,2] - cv[W,2]) + \
                                                  (m_data[i,5]*m_data[i,4] - m_data[i,4]*m_data[i,5])*(h_data[N,1] - h_data[S,1])/(cv[N,3] - cv[S,3]) \
                                                      )
        
        # Final step 
        t_dataC[i,1] = t_data[i,1] + derAVG[i,0]*dt

    return t_dataC