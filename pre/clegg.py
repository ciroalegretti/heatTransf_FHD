#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 01:47:56 2021

@author: alegretti
"""
import numpy as np


def clegg(magData,volNumb,cv,S,Le):
    """
    
    Solving Clegg equations for H field for a permanent magnet positioned externaly
    at the step face (no field before sudden expansion)
    
    """
    
    j = 1 
    b = S/2    # Clegg magnet height = 2b
    a = 1E+5     # Clegg magnet width = 2a
    
    x0 = Le
    y0 = S/2
    
    for i in range(volNumb):
        # if cv[i,2]>Le:
    
        y = cv[i,3] - y0  # x for clegg = y for bfs
        x = 0         # 2D field  
        z = cv[i,2] - x0
        
        
        # Hx in BFS coord system (Hz in cleeg equations)
        magData[i,3] = (j)*(np.arctan(((x+a)*(y+b))/(z*(((x+a)**2+(y+b)**2+z**2)**(0.5)))) \
                          + np.arctan(((x-a)*(y-b))/(z*(((x-a)**2+(y-b)**2+z**2)**(0.5))))\
                          - np.arctan(((x+a)*(y-b))/(z*(((x+a)**2+(y-b)**2+z**2)**(0.5))))\
                          - np.arctan(((x-a)*(y+b))/(z*(((x-a)**2+(y+b)**2+z**2)**(0.5)))))
              
        # Hy in BFS coord system (Hy in cleeg equations)
        magData[i,4] = (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z**2)**(0.5))/((x-a)+\
                       ((y-b)**2+(x-a)**2+z**2)**(0.5)))*(((x-a)+((y+b)**2+(x-a)**2\
                        +z**2)**(0.5))/((x+a)+((y+b)**2+(x+a)**2+z**2)**(0.5)))))
                                                              
    for i in range(volNumb):
        
        H0 = np.max(np.sqrt(magData[i,3]**2 + magData[i,4]**2))    
                    
    # Dimensionless fields                    
    magData[:,3] = magData[:,3]/H0
    magData[:,4] = magData[:,4]/H0    
                                     
    return magData,H0                                                                  
  