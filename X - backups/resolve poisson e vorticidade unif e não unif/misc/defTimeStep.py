#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:32:58 2021

@author: alegretti
"""
import numpy as np

def dtMin(Re,Pr,dtFac,x1,x3,y2,y3):
    """ Calculates minimum time step required"""
    
    dtPsi = dtFac*np.min([x1[0,-1] - x1[0,-2],x3[0,1] - x3[0,0],y2[0,0] - y2[1,0],y3[0,0] - y3[1,0]])**2         
    dtVort = (Re)*dtFac*np.min([x1[0,-1] - x1[0,-2],x3[0,1] - x3[0,0],y2[0,0] - y2[1,0],y3[0,0] - y3[1,0]])**2         
    dtTheta = (Re*Pr)*dtFac*np.min([x1[0,-1] - x1[0,-2],x3[0,1] - x3[0,0],y2[0,0] - y2[1,0],y3[0,0] - y3[1,0]])**2   
    
    dt = np.min([dtVort,dtTheta,dtPsi])    
    
    return dt