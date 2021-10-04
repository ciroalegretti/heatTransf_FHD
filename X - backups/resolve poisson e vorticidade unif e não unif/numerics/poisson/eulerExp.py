#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 20:21:44 2021

@author: alegretti
"""
from numba import jit
from misc import idNeighbors


@jit(nopython=True)
def poissonExplicit(data,volArr,volNodes,CV):

    dataPrev = data.copy()
    
    for i in range(len(volArr)):
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)

        a = (CV[E,2] - CV[W,2])/(CV[N,3] - CV[i,3])
        b = (CV[E,2] - CV[W,2])/(CV[i,3] - CV[S,3])
        c = (CV[N,3] - CV[S,3])/(CV[E,2] - CV[i,2])
        d = (CV[N,3] - CV[S,3])/(CV[i,2] - CV[W,2])
        
        data[i,3] = (1./(a + b + c + d))*(2.*dataPrev[i,4]*CV[i,1] + a*dataPrev[N,3] + b*dataPrev[S,3] + c*dataPrev[E,3] + d*dataPrev[W,3])
            
        
    return data