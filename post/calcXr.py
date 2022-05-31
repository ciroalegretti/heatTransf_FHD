#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:57:31 2021

@author: alegretti
"""
import numpy as np

def calcXr(edgeVol,volArr,fieldData,CVdata,Le,y1,S,nx3):
    # skipVol = 0
    xr = np.array(())
    # Xr = 0.0
    # while Xr == 0.0:   # Calculating only primary recirculation lenght
    for i in range(len(volArr)-nx3+1,len(volArr)-1):
        S = volArr[i,4]
        if S < 0:
            E = volArr[i,3]
            if fieldData[E,1]*fieldData[i,1] < 0:
                xr = np.append(xr , (CVdata[E,2] - fieldData[E,1]*(CVdata[E,2] - CVdata[i,2])/(fieldData[E,1] - fieldData[i,1]) - Le ))/S
                # Xr = ( CVdata[E,2] - fieldData[E,1]*(CVdata[E,2] - CVdata[i,2])/(fieldData[E,1] - fieldData[i,1]) - Le )/S  # Linear interpolation
                # Xr = ((fieldData[E,1]*CVdata[i,2] - fieldData[i,1]*CVdata[E,2])/(fieldData[E,1] - fieldData[i,1]) - Le)/S  # Linear interpolation

    # # Calling primary bubble reattachment point (others may have been identified)
    # if len(xr) == 0:    # none 
    #     xr1 = 0
    # elif len(xr) == 1:    # one
    #     xr1 = xr[0]
    # else:               # more than 1 recirculation zone
    #     xr1 = xr[1]

    # print(Xr)
    
    return xr