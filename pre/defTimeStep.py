#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:32:58 2021

@author: alegretti
"""
import numpy as np

def dtMin(Re,Pr,dtFac,CVdata):
    """ Calculates minimum time step required"""
    
    dtPsi = dtFac*np.min([np.min(CVdata[:,4]),np.min(CVdata[:,5])])**2            
    dtVort = (Re)*dtFac*np.min([np.min(CVdata[:,4]),np.min(CVdata[:,5])])**2            
    dtTheta = (Re*Pr)*dtFac*np.min([np.min(CVdata[:,4]),np.min(CVdata[:,5])])**2            
    
    dt = np.min([dtVort,dtTheta,dtPsi])    
    
    return dt