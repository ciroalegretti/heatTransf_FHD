#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 22:46:38 2021

@author: alegretti
"""
from numba import jit
from misc import idNeighbors

@jit(nopython=True)
def calculateUandV(data,volArr,controlVol):
    
    for i in range(len(volArr)):
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)
        
        data[i,1] =  (data[N,3] - data[S,3])/(controlVol[N,3] - controlVol[S,3])
        data[i,2] = -(data[E,3] - data[W,3])/(controlVol[E,2] - controlVol[W,2])
        
    return data
