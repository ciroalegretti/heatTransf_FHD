#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:57:31 2021

@author: alegretti
"""

def calcXr(edgeVol,volArr,fieldData,CVdata,Le):
    Xr = 0.0
    while Xr == 0.0:   # Calculating only primary recirculation lenght
        for i in range(edgeVol,len(volArr)):
            S = volArr[i,4]
            if S < 0:
                E = volArr[i,3]
                if fieldData[E,1]*fieldData[i,1] < 0:
                    Xr = CVdata[E,2] - fieldData[E,1]*(CVdata[E,2] - CVdata[i,2])/(fieldData[E,1] - fieldData[i,1]) - Le

    return Xr