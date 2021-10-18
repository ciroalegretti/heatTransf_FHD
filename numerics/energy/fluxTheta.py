#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 15:37:39 2021

@author: alegretti
"""
from numba import jit,prange
from pre import idNeighbors

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
