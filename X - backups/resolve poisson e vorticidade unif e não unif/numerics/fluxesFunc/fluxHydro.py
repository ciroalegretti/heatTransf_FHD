#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 03:22:33 2021

@author: alegretti
"""
from numba import jit
from main_func import idNeighbors

@jit(nopython=True)
def fluxHydro_C(flux,h_data,wf,cv,volArr):

    k = len(flux)
    
    for i in range(k):
        W,N,E,S = idNeighbors(volArr,i)
        
        # # Flux in E volume direction
        # flux[i,0] = + ((1 - wf[i,1])*h_data[E,1] - wf[i,1]*h_data[i,1])*((1 - wf[i,1])*h_data[E,4] - wf[i,1]*h_data[i,4]) 
        # # Flux in N volume direction
        # flux[i,1] = + ((1 - wf[i,2])*h_data[N,2] - wf[i,2]*h_data[i,2])*((1 - wf[i,2])*h_data[N,4] - wf[i,2]*h_data[i,4]) 
        # # Flux in W volume direction
        # flux[i,2] = + (wf[i,3]*h_data[i,1] - (1 - wf[i,3])*h_data[W,1])*(wf[i,3]*h_data[i,4] - (1 - wf[i,3])*h_data[W,4]) 
        # # Flux in S volume direction
        # flux[i,3] = + (wf[i,4]*h_data[i,2] - (1 - wf[i,4])*h_data[S,2])*(wf[i,4]*h_data[i,4] - (1 - wf[i,4])*h_data[S,4])

        """Teste"""
        
        # Flux in E volume direction
        flux[i,0] = h_data[i,1]*((1 - wf[i,1])*h_data[E,4] + wf[i,1]*h_data[i,4]) \
                    # - (1/c)*((1 - wf[i,1])*h_data[E,4] - wf[i,1]*h_data[i,4])/(cv[E,2] - cv[i,2])
        # Flux in N volume direction
        flux[i,1] = h_data[i,2]*((1 - wf[i,2])*h_data[N,4] + wf[i,2]*h_data[i,4]) \
                    # - (1/c)*((1 - wf[i,2])*h_data[N,4] - wf[i,2]*h_data[i,4])/(cv[N,3] - cv[i,3])
        # Flux in W volume direction
        flux[i,2] = h_data[i,1]*(wf[i,3]*h_data[i,4] + (1 - wf[i,3])*h_data[W,4]) \
                    # + (1/c)*(wf[i,3]*h_data[i,4] - (1 - wf[i,3])*h_data[W,4])/(cv[i,2] - cv[W,2])
        # Flux in S volume direction
        flux[i,3] = h_data[i,2]*(wf[i,4]*h_data[i,4] + (1 - wf[i,4])*h_data[S,4]) \
                    # + (1/c)*(wf[i,4]*h_data[i,4] - (1 - wf[i,4])*h_data[S,4])/(cv[i,3] - cv[S,3])

    return flux


@jit(nopython=True)
def fluxHydro_D(flux,h_data,c,wf,cv,volArr):

    k = len(flux)
    
    for i in range(k):
        W,N,E,S = idNeighbors(volArr,i)
        
        # #Flux in E volume direction
        # flux[i,0] = + (1./c)*((1 - wf[i,1])*h_data[E,4] - wf[i,1]*h_data[i,4])/(cv[E,2] - cv[i,2])
        # # Flux in N volume direction
        # flux[i,1] = + (1./c)*((1 - wf[i,2])*h_data[N,4] - wf[i,2]*h_data[i,4])/(cv[N,3] - cv[i,3])
        # # Flux in W volume direction
        # flux[i,2] = + (1./c)*(wf[i,3]*h_data[i,4] - (1 - wf[i,3])*h_data[W,4])/(cv[i,2] - cv[W,2])
        # # Flux in S volume direction
        # flux[i,3] = + (1./c)*(wf[i,4]*h_data[i,4] - (1 - wf[i,4])*h_data[S,4])/(cv[i,3] - cv[S,3])

        # Flux in E volume direction
        flux[i,0] = + (1./c)*(h_data[E,4] - h_data[i,4])/(cv[E,2] - cv[i,2])
        # Flux in N volume direction
        flux[i,1] = + (1./c)*(h_data[N,4] - h_data[i,4])/(cv[N,3] - cv[i,3])
        # Flux in W volume direction
        flux[i,2] = + (1./c)*(h_data[i,4] - h_data[W,4])/(cv[i,2] - cv[W,2])
        # Flux in S volume direction
        flux[i,3] = + (1./c)*(h_data[i,4] - h_data[S,4])/(cv[i,3] - cv[S,3])

        """Teste"""
        
        # # Flux in E volume direction
        # flux[i,1] = h_data[i,1]*((1 - wf[i,1])*h_data[E,4] - wf[i,1]*h_data[i,4]) \
        #             # - (1/c)*((1 - wf[i,1])*h_data[E,4] - wf[i,1]*h_data[i,4])/(cv[E,2] - cv[i,2])
        # # Flux in N volume direction
        # flux[i,2] = h_data[i,2]*((1 - wf[i,2])*h_data[N,4] - wf[i,2]*h_data[i,4]) \
        #             # - (1/c)*((1 - wf[i,2])*h_data[N,4] - wf[i,2]*h_data[i,4])/(cv[N,3] - cv[i,3])
        # # Flux in W volume direction
        # flux[i,3] = - h_data[i,1]*(wf[i,3]*h_data[i,4] - (1 - wf[i,3])*h_data[W,4]) \
        #             # + (1/c)*(wf[i,3]*h_data[i,4] - (1 - wf[i,3])*h_data[W,4])/(cv[i,2] - cv[W,2])
        # # Flux in S volume direction
        # flux[i,4] = - h_data[i,2]*(wf[i,4]*h_data[i,4] - (1 - wf[i,4])*h_data[S,4]) \
        #             # + (1/c)*(wf[i,4]*h_data[i,4] - (1 - wf[i,4])*h_data[S,4])/(cv[i,3] - cv[S,3])

    return flux