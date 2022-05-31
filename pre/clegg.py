#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 01:47:56 2021

@author: alegretti
"""
import numpy as np

def cleggStepFace(magData,volNumb,cv,S,Le):
    """
    
    Solving Clegg equations for H field for a permanent magnet positioned externaly
    at the step face (no field before sudden expansion)
    
    """
    
    j = 1.2 
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
                                     
    return magData[:,3],magData[:,4]     

def cleggStepBW(magData,volNumb,cv,S,Le,Ls):
    """
    
    Solving Clegg equations for H field for a permanent magnet positioned externaly
    at the step face (no field before sudden expansion)
    
    """
    
    j = 1.2 
    b = Ls/6     # Clegg magnet height = 2b
    a = 1E+5     # Clegg magnet width = 2a
    
    x0 = 3*Le/2#(Ls*2/3 + Le)
    y0 = 0.0
    
    for i in range(volNumb):
        # if cv[i,2]>Le:
    
        y = cv[i,2] - x0  # x for clegg = y for bfs
        x = 0         # 2D field  
        z = cv[i,3] - y0
        
        
        # Hx in BFS coord system (Hz in cleeg equations)
        magData[i,4] = (j)*(np.arctan(((x+a)*(y+b))/(z*(((x+a)**2+(y+b)**2+z**2)**(0.5)))) \
                          + np.arctan(((x-a)*(y-b))/(z*(((x-a)**2+(y-b)**2+z**2)**(0.5))))\
                          - np.arctan(((x+a)*(y-b))/(z*(((x+a)**2+(y-b)**2+z**2)**(0.5))))\
                          - np.arctan(((x-a)*(y+b))/(z*(((x-a)**2+(y+b)**2+z**2)**(0.5)))))
              
        # Hy in BFS coord system (Hy in cleeg equations)
        magData[i,3] = (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z**2)**(0.5))/((x-a)+\
                       ((y-b)**2+(x-a)**2+z**2)**(0.5)))*(((x-a)+((y+b)**2+(x-a)**2\
                        +z**2)**(0.5))/((x+a)+((y+b)**2+(x+a)**2+z**2)**(0.5)))))
                                                              
    for i in range(volNumb):
        
        H0 = np.max(np.sqrt(magData[i,3]**2 + magData[i,4]**2))    
                    
    # Dimensionless fields                    
    magData[:,3] = magData[:,3]/H0
    magData[:,4] = magData[:,4]/H0    
                                     
    return magData[:,3],magData[:,4]       

def cleggPP(magData,volNumb,cv,h,Le):
    """
    
    Solving Clegg equations for H field for a permanent magnet positioned at the bottom wall of parallel plates domain
    
    """
    H0 = magData[:,:2].copy()
    j = 1.2/(4*np.pi*10**-7)
    a = 1E+5#Le/8    # Clegg magnet width = 2a
    b = Le #1E+5     # Clegg magnet lenght = 2b
    c = h        # distance between opposite faces
    
    x0 = Le/2#Le/2
    y0 = 0.0
    
    for i in range(volNumb):
        # if cv[i,2]>Le:
    
        y = cv[i,2] - x0  # x for clegg = x for PP
        x = 0         # 2D field  
        z = cv[i,3] - y0 # z for clegg = y for PP
        z2 = z + c
        
        
        # Hy in PP coord system (Hx in cleeg equations)
        magData[i,4] = (j)*(np.arctan(((x+a)*(y+b))/(z*(((x+a)**2+(y+b)**2+z**2)**(0.5)))) \
                          + np.arctan(((x-a)*(y-b))/(z*(((x-a)**2+(y-b)**2+z**2)**(0.5))))\
                          - np.arctan(((x+a)*(y-b))/(z*(((x+a)**2+(y-b)**2+z**2)**(0.5))))\
                          - np.arctan(((x-a)*(y+b))/(z*(((x-a)**2+(y+b)**2+z**2)**(0.5))))) \
                      - (j)*(np.arctan(((x+a)*(y+b))/(z2*(((x+a)**2+(y+b)**2+z2**2)**(0.5)))) \
                           + np.arctan(((x-a)*(y-b))/(z2*(((x-a)**2+(y-b)**2+z2**2)**(0.5))))\
                           - np.arctan(((x+a)*(y-b))/(z2*(((x+a)**2+(y-b)**2+z2**2)**(0.5))))\
                           - np.arctan(((x-a)*(y+b))/(z2*(((x-a)**2+(y+b)**2+z2**2)**(0.5)))))
              
        # Hx in PP coord system (Hy in cleeg equations)
        magData[i,3] = (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z**2)**(0.5))/((x-a)+\
                       ((y-b)**2+(x-a)**2+z**2)**(0.5)))*(((x-a)+((y+b)**2+(x-a)**2\
                        +z**2)**(0.5))/((x+a)+((y+b)**2+(x+a)**2+z**2)**(0.5))))) \
                       - (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z2**2)**(0.5))/((x-a)+\
                         ((y-b)**2+(x-a)**2+z2**2)**(0.5)))*(((x-a)+((y+b)**2+(x-a)**2\
                          +z2**2)**(0.5))/((x+a)+((y+b)**2+(x+a)**2+z2**2)**(0.5)))))
                                                              
    for i in range(volNumb):
        
        H0[i,1] = np.sqrt(magData[i,3]**2 + magData[i,4]**2)
                    
    H0max = np.max(H0[:,1])
    # Dimensionless fields                    
    magData[:,3] = magData[:,3]/H0max
    magData[:,4] = magData[:,4]/H0max    
                                     
    return magData[:,3],magData[:,4]      

def cleggLDC(magData,volNumb,cv,h,Le):
    """
    
    Solving Clegg equations for H field for a permanent magnet positioned at the bottom wall of parallel plates domain
    
    """
    H0 = magData[:,:2].copy()
    j = 1.2/(4*np.pi*10**-7)
    a = 1E+5#Le/8    # Clegg magnet width = 2a
    b = Le/10 #1E+5     # Clegg magnet lenght = 2b
    c = h        # distance between opposite faces
    
    x0 = Le/2#Le/2
    y0 = 0.0
    
    for i in range(volNumb):
        # if cv[i,2]>Le:
    
        y = cv[i,2] - x0  # x for clegg = x for PP
        x = 0         # 2D field  
        z = cv[i,3] - y0 # z for clegg = y for PP
        z2 = z + c
        
        
        # Hy in PP coord system (Hx in cleeg equations)
        magData[i,4] = (j)*(np.arctan(((x+a)*(y+b))/(z*(((x+a)**2+(y+b)**2+z**2)**(0.5)))) \
                          + np.arctan(((x-a)*(y-b))/(z*(((x-a)**2+(y-b)**2+z**2)**(0.5))))\
                          - np.arctan(((x+a)*(y-b))/(z*(((x+a)**2+(y-b)**2+z**2)**(0.5))))\
                          - np.arctan(((x-a)*(y+b))/(z*(((x-a)**2+(y+b)**2+z**2)**(0.5))))) \
                      - (j)*(np.arctan(((x+a)*(y+b))/(z2*(((x+a)**2+(y+b)**2+z2**2)**(0.5)))) \
                           + np.arctan(((x-a)*(y-b))/(z2*(((x-a)**2+(y-b)**2+z2**2)**(0.5))))\
                           - np.arctan(((x+a)*(y-b))/(z2*(((x+a)**2+(y-b)**2+z2**2)**(0.5))))\
                           - np.arctan(((x-a)*(y+b))/(z2*(((x-a)**2+(y+b)**2+z2**2)**(0.5)))))
              
        # Hx in PP coord system (Hy in cleeg equations)
        magData[i,3] = (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z**2)**(0.5))/((x-a)+\
                       ((y-b)**2+(x-a)**2+z**2)**(0.5)))*(((x-a)+((y+b)**2+(x-a)**2\
                        +z**2)**(0.5))/((x+a)+((y+b)**2+(x+a)**2+z**2)**(0.5))))) \
                       - (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z2**2)**(0.5))/((x-a)+\
                         ((y-b)**2+(x-a)**2+z2**2)**(0.5)))*(((x-a)+((y+b)**2+(x-a)**2\
                          +z2**2)**(0.5))/((x+a)+((y+b)**2+(x+a)**2+z2**2)**(0.5)))))
                                                              
    for i in range(volNumb):
        
        H0[i,1] = np.sqrt(magData[i,3]**2 + magData[i,4]**2)
                    
    H0max = np.max(H0[:,1])
    # Dimensionless fields                    
    magData[:,3] = magData[:,3]/H0max
    magData[:,4] = magData[:,4]/H0max    
                                     
    return magData[:,3],magData[:,4]      


def externalField(field_type,magData,volNumb,cv,x0,y0,b,Lm,n):
# =============================================================================
#     
#     
#     Solving 2D Clegg equations for H field for a permanent magnet with face centroid positioned at x = x0, y = y0
#     
#     
# =============================================================================
    ex = n[0]
    ey = n[1]
    
    if field_type == 'unif':
        ### Uniform applied field
        magData[:,3] = ex*1 
        magData[:,4] = ey*1 
        
    elif field_type == 'clegg':
    
        j = 1.2/(4*np.pi*10**-7)
        a = 1E+5     # Infinite width (2D field)
        
        for i in range(volNumb):
            x = 0
            y = (cv[i,3] - y0)*ex + (cv[i,2] - x0)*ey   
            z = (cv[i,2] - x0)*ex + (cv[i,3] - y0)*ey   
            z2 = z + Lm
            
            magData[i,3] = ey*((j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z**2)**0.5)/((x-a)+((y-b)**2+(x-a)**2+z**2)**0.5))*       \
                                           (((x-a)+((y+b)**2+(x-a)**2+z**2)**0.5)/((x+a)+((y+b)**2+(x+a)**2+z**2)**0.5))))      \
                             - (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z2**2)**0.5)/((x-a)+((y-b)**2+(x-a)**2+z2**2)**0.5))*    \
                                           (((x-a)+((y+b)**2+(x-a)**2+z2**2)**0.5)/((x+a)+((y+b)**2+(x+a)**2+z2**2)**0.5))))) \
                         + ex*((j)*(np.arctan(((x+a)*(y+b))/(z*(((x+a)**2+(y+b)**2+z**2)**0.5)))    \
                                  + np.arctan(((x-a)*(y-b))/(z*(((x-a)**2+(y-b)**2+z**2)**0.5)))    \
                                  - np.arctan(((x+a)*(y-b))/(z*(((x+a)**2+(y-b)**2+z**2)**0.5)))    \
                                  - np.arctan(((x-a)*(y+b))/(z*(((x-a)**2+(y+b)**2+z**2)**0.5))))   \
                             - (j)*(np.arctan(((x+a)*(y+b))/(z2*(((x+a)**2+(y+b)**2+z2**2)**0.5)))  \
                                  + np.arctan(((x-a)*(y-b))/(z2*(((x-a)**2+(y-b)**2+z2**2)**0.5)))  \
                                  - np.arctan(((x+a)*(y-b))/(z2*(((x+a)**2+(y-b)**2+z2**2)**0.5)))  \
                                  - np.arctan(((x-a)*(y+b))/(z2*(((x-a)**2+(y+b)**2+z2**2)**0.5)))) )
    
    
            magData[i,4] = ex*((j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z**2)**0.5)/((x-a)+((y-b)**2+(x-a)**2+z**2)**0.5))*       \
                                           (((x-a)+((y+b)**2+(x-a)**2+z**2)**0.5)/((x+a)+((y+b)**2+(x+a)**2+z**2)**0.5))))      \
                             - (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z2**2)**0.5)/((x-a)+((y-b)**2+(x-a)**2+z2**2)**0.5))*    \
                                           (((x-a)+((y+b)**2+(x-a)**2+z2**2)**0.5)/((x+a)+((y+b)**2+(x+a)**2+z2**2)**0.5))))) \
                         + ey*((j)*(np.arctan(((x+a)*(y+b))/(z*(((x+a)**2+(y+b)**2+z**2)**0.5)))    \
                                  + np.arctan(((x-a)*(y-b))/(z*(((x-a)**2+(y-b)**2+z**2)**0.5)))    \
                                  - np.arctan(((x+a)*(y-b))/(z*(((x+a)**2+(y-b)**2+z**2)**0.5)))    \
                                  - np.arctan(((x-a)*(y+b))/(z*(((x-a)**2+(y+b)**2+z**2)**0.5))))   \
                             - (j)*(np.arctan(((x+a)*(y+b))/(z2*(((x+a)**2+(y+b)**2+z2**2)**0.5)))  \
                                  + np.arctan(((x-a)*(y-b))/(z2*(((x-a)**2+(y-b)**2+z2**2)**0.5)))  \
                                  - np.arctan(((x+a)*(y-b))/(z2*(((x+a)**2+(y-b)**2+z2**2)**0.5)))  \
                                  - np.arctan(((x-a)*(y+b))/(z2*(((x-a)**2+(y+b)**2+z2**2)**0.5)))) )
                
        
    
    H0 = np.zeros((volNumb))
    for i in range(volNumb):
        H0[i] = np.sqrt(magData[i,3]**2 + magData[i,4]**2)
        
    H0max = np.max(H0)
                    
    # Dimensionless fields                    
    magData[:,3] = magData[:,3]/H0max
    magData[:,4] = magData[:,4]/H0max
                                     
    return magData[:,3],magData[:,4],H0