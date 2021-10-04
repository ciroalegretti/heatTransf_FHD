#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:41:34 2021

@author: alegretti
"""
from numba import jit

@jit(nopython=True)
def idNeighbors(volList,N):
    """
    Recieves a volume number and returns the numbers of its four neighbors
    """
    w = volList[N,1]
    n = volList[N,2]
    e = volList[N,3]
    s = volList[N,4]
    
    return w,n,e,s

@jit(nopython=True)
def idDiagonalVol(volList,N):
    """
    Recieves a volume number and returns the numbers of its four diagonal neighbors 
    """
    w = volList[N,1]
    nw = volList[w,2]
    n = volList[N,2]
    ne = volList[n,3]
    e = volList[N,3]
    se = volList[e,4]
    s = volList[N,4]
    sw = volList[s,1]
    
    return nw,ne,se,sw