#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 17:27:46 2021

@author: alegretti
"""
from numba import jit
from pre import idNeighbors

@jit(nopython=True)
# """________________________x-component Magnetic Equation________________________"""
def ftcsMx(data,magData,volArr,cv,volNodes,nodesCoord,dt,phi,pe):
    
    magDataP = magData.copy()
    
    for i in range(len(volArr)):
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)
        
        Upos = max(data[i,1],0)
        Uneg = min(data[i,1],0)
        Vpos = max(data[i,2],0)
        Vneg = min(data[i,2],0)
        MxFoX = (magDataP[E,5] - magDataP[i,5])/(cv[E,2] - cv[i,2])
        MxBaX = (magDataP[i,5] - magDataP[W,5])/(cv[i,2] - cv[W,2])
        MxFoY = (magDataP[N,5] - magDataP[i,5])/(cv[N,3] - cv[i,3])
        MxBaY = (magDataP[i,5] - magDataP[S,5])/(cv[i,3] - cv[S,3])
        
        magData[i,5] = magDataP[i,5] + dt*( \
                                      - (Upos*MxBaX + Uneg*MxFoX)/2 \
                                      - (Vpos*MxBaY + Vneg*MxFoY)/2 \
                                      - data[i,4]*magDataP[i,6]/2. + (magDataP[i,3]*magDataP[i,6]**2 - (magDataP[i,5]*magDataP[i,6]*magDataP[i,4]))*3*magData[i,-1]/(4*pe) \
                                      + (magDataP[i,1] - magDataP[i,5])/pe
                                          )
                                      
    return magData

@jit(nopython=True)
# """________________________y-component Magnetic Equation________________________"""
def ftcsMy(data,magData,volArr,cv,volNodes,nodesCoord,dt,phi,pe):
    
    magDataP = magData.copy()
    
    for i in range(len(volArr)):
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)
        
        Upos = max(data[i,1],0)
        Uneg = min(data[i,1],0)
        Vpos = max(data[i,2],0)
        Vneg = min(data[i,2],0)
        MyFoX = (magDataP[E,6] - magDataP[i,6])/(cv[E,2] - cv[i,2])
        MyBaX = (magDataP[i,6] - magDataP[W,6])/(cv[i,2] - cv[W,2])
        MyFoY = (magDataP[N,6] - magDataP[i,6])/(cv[N,3] - cv[i,3])
        MyBaY = (magDataP[i,6] - magDataP[S,6])/(cv[i,3] - cv[S,3])
        
        magData[i,6] = magDataP[i,6] + dt*( \
                                      - (Upos*MyBaX + Uneg*MyFoX)/2 \
                                      - (Vpos*MyBaY + Vneg*MyFoY)/2 \
                                      + data[i,4]*magDataP[i,5]/2. + (magDataP[i,4]*(magDataP[i,5]**2) - magDataP[i,5]*magDataP[i,6]*magDataP[i,4] )*3*magData[i,-1]/(4*pe)   \
                                      + (magDataP[i,2] - magDataP[i,6])/pe
                                          )
                                      
    return magData