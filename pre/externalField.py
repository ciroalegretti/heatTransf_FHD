 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 01:47:56 2021

@author: alegretti
"""
import numpy as np

def inputField(magnet):
# =============================================================================
#     
#     Returns unity vector 'eh' normal to the magnet
#     
# =============================================================================

    if magnet == 'horiz':
        eh = [1,0]
    elif magnet == 'vert':
        eh = [0,1]
    else:
        print('No orientation defined')

    return eh

def getCoordLDC(eh,L):
    
# =============================================================================
#
#     Recieves 'eh' and returns (depending on field orientation):
#    
#           x0 - x coord of magnet face centroid
#           y0 - y coord of magnet face centroid
#           b  - half lenght of magnet face (lenght = 2b by Clegg)
#           Lm - magnet width (distance between oposite faces)
#
# =============================================================================
    
    if eh == [1,0]:
        orient = 'horiz'
    elif eh == [0,1]:
        orient = 'vert'

    if orient == 'horiz':   # Horizontal
        x0 = 0
        y0 = L/2
        b = L/6
        Lm = L
    
    elif orient == 'vert': # Vertical
        x0 = L/2	 
        y0 = 0
        b = L/6
        Lm = L
        
    return x0,y0,b,Lm

def getCoordPP(eh,L,h,dist,ratio):
    
# =============================================================================
#
#     x0 - x coord of magnet face centroid
#     y0 - y coord of magnet face centroid
#     b  - half lenght of magnet face (lenght = 2b by Clegg)
#     Lm - magnet width (distance between oposite faces)
#   dist - vertical distance between the magnet and the bottom wall
#  ratio - ratio between the lenghts of the channel by the half of the magnet's (L/b)
#
# =============================================================================
    
    if eh == [1,0]:
        orient = 'horiz'
    elif eh == [0,1]:
        orient = 'vert'
     
        if orient == 'horiz':   # Horizontal
            h = 0.5
            x0 = 0
            y0 = h/2
            b = h/2
            Lm = h        
        
        elif orient == 'vert': # Vertical
            x0 = 0.5*L	 
            y0 = 0 - dist
            b = L/(2*ratio)
            Lm = L

        return x0,y0,b,Lm    

def getCoordBFS(eh,Le,S,dist,ratio):
    
# =============================================================================
#
#     x0 - x coord of magnet face centroid
#     y0 - y coord of magnet face centroid
#     b  - half lenght of magnet face 
#     Lm - magnet width (distance between oposite faces)
#
# =============================================================================
    
    if eh == [1,0]:
        orient = '1'
    if eh == [0,1]:
        orient = '2'


    if orient == '1':   # Horizontal at step face
        x0 = Le - dist
        y0 = S/2
        b = S/2
        Lm = S
    
    if orient == '2': # Vertical right after sudden expansion
        b = S/(2*ratio)
        x0 = Le + b
        y0 = 0 - dist
        Lm = S      
        
    return x0,y0,b,Lm            
        
            

def externalField(field_sty,eh,magData,volNumb,cv,x0,y0,b,Lm):
# =============================================================================
#     
#     Field type:
#           
#       clegg - analytical solution for a permanent magnet (x0,y0,b,Lm)
#       unif  - uniform field
#     
# =============================================================================
    
    # x0,y0,b,Lm = getCoord(model,field_sty,eh,L,Le,S)

    ex = eh[0]
    ey = eh[1]
    
    if field_sty == 'unif':
        ### Uniform applied field
        magData[:,3] = ex*1 
        magData[:,4] = ey*1 
        
    elif field_sty == 'clegg':
        ### Clegg solution
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
                
    elif field_sty == 'gradUnif':
        ### 1d uniform gradient field
        height = np.round(np.max(cv[:volNumb,3]),1)
        for i in range(volNumb):
            magData[i,3] = (1 - (cv[i,2]/height))*ex 
            magData[i,4] = (1 - (cv[i,3]/height))*ey         
    
    else:
        print('bug field type')
    
    H0 = np.zeros((volNumb))
    for i in range(volNumb):
        H0[i] = np.sqrt(magData[i,3]**2 + magData[i,4]**2)
        
    H0max = np.max(H0)
                    
    # Dimensionless fields                    
    magData[:,3] = magData[:,3]/H0max
    magData[:,4] = magData[:,4]/H0max
    
    H0 = H0/H0max
                                     
    return magData[:,3],magData[:,4],H0



# def cleggStepFace(magData,volNumb,cv,S,Le):
#     """
    
#     Solving Clegg equations for H field for a permanent magnet positioned externaly
#     at the step face (no field before sudden expansion)
    
#     """
    
#     j = 1.2 
#     b = S/2    # Clegg magnet height = 2b
#     a = 1E+5     # Clegg magnet width = 2a
    
#     x0 = Le
#     y0 = S/2
    
#     for i in range(volNumb):
#         # if cv[i,2]>Le:
    
#         y = cv[i,3] - y0  # x for clegg = y for bfs
#         x = 0         # 2D field  
#         z = cv[i,2] - x0
        
        
#         # Hx in BFS coord system (Hz in cleeg equations)
#         magData[i,3] = (j)*(np.arctan(((x+a)*(y+b))/(z*(((x+a)**2+(y+b)**2+z**2)**(0.5)))) \
#                           + np.arctan(((x-a)*(y-b))/(z*(((x-a)**2+(y-b)**2+z**2)**(0.5))))\
#                           - np.arctan(((x+a)*(y-b))/(z*(((x+a)**2+(y-b)**2+z**2)**(0.5))))\
#                           - np.arctan(((x-a)*(y+b))/(z*(((x-a)**2+(y+b)**2+z**2)**(0.5)))))
              
#         # Hy in BFS coord system (Hy in cleeg equations)
#         magData[i,4] = (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z**2)**(0.5))/((x-a)+\
#                        ((y-b)**2+(x-a)**2+z**2)**(0.5)))*(((x-a)+((y+b)**2+(x-a)**2\
#                         +z**2)**(0.5))/((x+a)+((y+b)**2+(x+a)**2+z**2)**(0.5)))))
                                                              
#     for i in range(volNumb):
        
#         H0 = np.max(np.sqrt(magData[i,3]**2 + magData[i,4]**2))    
                    
#     # Dimensionless fields                    
#     magData[:,3] = magData[:,3]/H0
#     magData[:,4] = magData[:,4]/H0    
                                     
#     return magData[:,3],magData[:,4]     

# def cleggStepBW(magData,volNumb,cv,S,Le,Ls):
#     """
    
#     Solving Clegg equations for H field for a permanent magnet positioned externaly
#     at the step face (no field before sudden expansion)
    
#     """
    
#     j = 1.2 
#     b = Ls/6     # Clegg magnet height = 2b
#     a = 1E+5     # Clegg magnet width = 2a
    
#     x0 = 3*Le/2#(Ls*2/3 + Le)
#     y0 = 0.0
    
#     for i in range(volNumb):
#         # if cv[i,2]>Le:
    
#         y = cv[i,2] - x0  # x for clegg = y for bfs
#         x = 0         # 2D field  
#         z = cv[i,3] - y0
        
        
#         # Hx in BFS coord system (Hz in cleeg equations)
#         magData[i,4] = (j)*(np.arctan(((x+a)*(y+b))/(z*(((x+a)**2+(y+b)**2+z**2)**(0.5)))) \
#                           + np.arctan(((x-a)*(y-b))/(z*(((x-a)**2+(y-b)**2+z**2)**(0.5))))\
#                           - np.arctan(((x+a)*(y-b))/(z*(((x+a)**2+(y-b)**2+z**2)**(0.5))))\
#                           - np.arctan(((x-a)*(y+b))/(z*(((x-a)**2+(y+b)**2+z**2)**(0.5)))))
              
#         # Hy in BFS coord system (Hy in cleeg equations)
#         magData[i,3] = (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z**2)**(0.5))/((x-a)+\
#                        ((y-b)**2+(x-a)**2+z**2)**(0.5)))*(((x-a)+((y+b)**2+(x-a)**2\
#                         +z**2)**(0.5))/((x+a)+((y+b)**2+(x+a)**2+z**2)**(0.5)))))
                                                              
#     for i in range(volNumb):
        
#         H0 = np.max(np.sqrt(magData[i,3]**2 + magData[i,4]**2))    
                    
#     # Dimensionless fields                    
#     magData[:,3] = magData[:,3]/H0
#     magData[:,4] = magData[:,4]/H0    
                                     
#     return magData[:,3],magData[:,4]       

# def cleggPP(magData,volNumb,cv,h,Le):
#     """
    
#     Solving Clegg equations for H field for a permanent magnet positioned at the bottom wall of parallel plates domain
    
#     """
#     H0 = magData[:,:2].copy()
#     j = 1.2/(4*np.pi*10**-7)
#     a = 1E+5#Le/8    # Clegg magnet width = 2a
#     b = Le/10 #1E+5     # Clegg magnet lenght = 2b
#     c = h        # distance between opposite faces
    
#     x0 = Le/2#Le/2
#     y0 = 0.0
    
#     for i in range(volNumb):
#         # if cv[i,2]>Le:
    
#         y = cv[i,2] - x0  # x for clegg = x for PP
#         x = 0         # 2D field  
#         z = cv[i,3] - y0 # z for clegg = y for PP
#         z2 = z + c
        
        
#         # Hy in PP coord system (Hx in cleeg equations)
#         magData[i,4] = (j)*(np.arctan(((x+a)*(y+b))/(z*(((x+a)**2+(y+b)**2+z**2)**(0.5)))) \
#                           + np.arctan(((x-a)*(y-b))/(z*(((x-a)**2+(y-b)**2+z**2)**(0.5))))\
#                           - np.arctan(((x+a)*(y-b))/(z*(((x+a)**2+(y-b)**2+z**2)**(0.5))))\
#                           - np.arctan(((x-a)*(y+b))/(z*(((x-a)**2+(y+b)**2+z**2)**(0.5))))) \
#                       - (j)*(np.arctan(((x+a)*(y+b))/(z2*(((x+a)**2+(y+b)**2+z2**2)**(0.5)))) \
#                            + np.arctan(((x-a)*(y-b))/(z2*(((x-a)**2+(y-b)**2+z2**2)**(0.5))))\
#                            - np.arctan(((x+a)*(y-b))/(z2*(((x+a)**2+(y-b)**2+z2**2)**(0.5))))\
#                            - np.arctan(((x-a)*(y+b))/(z2*(((x-a)**2+(y+b)**2+z2**2)**(0.5)))))
              
#         # Hx in PP coord system (Hy in cleeg equations)
#         magData[i,3] = (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z**2)**(0.5))/((x-a)+\
#                        ((y-b)**2+(x-a)**2+z**2)**(0.5)))*(((x-a)+((y+b)**2+(x-a)**2\
#                         +z**2)**(0.5))/((x+a)+((y+b)**2+(x+a)**2+z**2)**(0.5))))) \
#                        - (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z2**2)**(0.5))/((x-a)+\
#                          ((y-b)**2+(x-a)**2+z2**2)**(0.5)))*(((x-a)+((y+b)**2+(x-a)**2\
#                           +z2**2)**(0.5))/((x+a)+((y+b)**2+(x+a)**2+z2**2)**(0.5)))))
                                                              
#     for i in range(volNumb):
        
#         H0[i,1] = np.sqrt(magData[i,3]**2 + magData[i,4]**2)
                    
#     H0max = np.max(H0[:,1])
#     # Dimensionless fields                    
#     magData[:,3] = magData[:,3]/H0max
#     magData[:,4] = magData[:,4]/H0max    
                                     
#     return magData[:,3],magData[:,4]      

# def cleggLDC(magData,volNumb,cv,h,Le):
#     """
    
#     Solving Clegg equations for H field for a permanent magnet positioned at the bottom wall of parallel plates domain
    
#     """
#     H0 = magData[:,:2].copy()
#     j = 1.2/(4*np.pi*10**-7)
#     a = 1E+5#Le/8    # Clegg magnet width = 2a
#     b = Le/10 #1E+5     # Clegg magnet lenght = 2b
#     c = h        # distance between opposite faces
    
#     x0 = Le/2#Le/2
#     y0 = 0.0
    
#     for i in range(volNumb):
#         # if cv[i,2]>Le:
    
#         y = cv[i,2] - x0  # x for clegg = x for PP
#         x = 0         # 2D field  
#         z = cv[i,3] - y0 # z for clegg = y for PP
#         z2 = z + c
        
        
#         # Hy in PP coord system (Hx in cleeg equations)
#         magData[i,4] = (j)*(np.arctan(((x+a)*(y+b))/(z*(((x+a)**2+(y+b)**2+z**2)**(0.5)))) \
#                           + np.arctan(((x-a)*(y-b))/(z*(((x-a)**2+(y-b)**2+z**2)**(0.5))))\
#                           - np.arctan(((x+a)*(y-b))/(z*(((x+a)**2+(y-b)**2+z**2)**(0.5))))\
#                           - np.arctan(((x-a)*(y+b))/(z*(((x-a)**2+(y+b)**2+z**2)**(0.5))))) \
#                       - (j)*(np.arctan(((x+a)*(y+b))/(z2*(((x+a)**2+(y+b)**2+z2**2)**(0.5)))) \
#                            + np.arctan(((x-a)*(y-b))/(z2*(((x-a)**2+(y-b)**2+z2**2)**(0.5))))\
#                            - np.arctan(((x+a)*(y-b))/(z2*(((x+a)**2+(y-b)**2+z2**2)**(0.5))))\
#                            - np.arctan(((x-a)*(y+b))/(z2*(((x-a)**2+(y+b)**2+z2**2)**(0.5)))))
              
#         # Hx in PP coord system (Hy in cleeg equations)
#         magData[i,3] = (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z**2)**(0.5))/((x-a)+\
#                        ((y-b)**2+(x-a)**2+z**2)**(0.5)))*(((x-a)+((y+b)**2+(x-a)**2\
#                         +z**2)**(0.5))/((x+a)+((y+b)**2+(x+a)**2+z**2)**(0.5))))) \
#                        - (j)*(np.log((((x+a)+((y-b)**2+(x+a)**2+z2**2)**(0.5))/((x-a)+\
#                          ((y-b)**2+(x-a)**2+z2**2)**(0.5)))*(((x-a)+((y+b)**2+(x-a)**2\
#                           +z2**2)**(0.5))/((x+a)+((y+b)**2+(x+a)**2+z2**2)**(0.5)))))
                                                              
#     for i in range(volNumb):
        
#         H0[i,1] = np.sqrt(magData[i,3]**2 + magData[i,4]**2)
                    
#     H0max = np.max(H0[:,1])
#     # Dimensionless fields                    
#     magData[:,3] = magData[:,3]/H0max
#     magData[:,4] = magData[:,4]/H0max    
                                     
#     return magData[:,3],magData[:,4]      

