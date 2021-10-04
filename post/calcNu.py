#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:26:00 2021

@author: alegretti
"""
import numpy as np

def calcNu(H,nx3,ny2,ny3,x3,Le,data,tData,cv,volArr,Ls,volNumb):
    import matplotlib.pyplot as plt
    
    print('')
    print('Calculating heat flux at the bottom wall and output parameters!')    
    
    x3 = np.round(x3,decimals=3)
    cv = np.round(cv,decimals=5)
    
    x_cv = x3[0,:] + (x3[0,1] - x3[0,0])/2   # every cv centroid x-coord after the sudden expansion
    x_cv = x_cv[:-1]                                   # number of volumes = nodes - 1
    x_cv = np.round(x_cv,decimals=3)
    
    # sliceVol = np.zeros((ny2 + ny3 + 1,len(x_cv)))         # 2D representation of volumes arrangement
    Tm = np.zeros((len(x_cv)))                           # Array of average temperatures Tm = Tm(x)
    wallVol = np.zeros((nx3 - 1))
    ghostWall = wallVol.copy()
    volSlice = np.zeros((ny2 + ny3 + 1,len(x_cv)))           # "+1" instead "-1" in order to include two ghosts
 
    u = np.zeros((ny2 + ny3 + 1))
    theta = u.copy()
    uTheta = u.copy()                   # product between u and theta for integration
    nu_x = x_cv.copy()
    thetaWall = x_cv.copy()
    theta1 = thetaWall.copy()           # border between first two volumes from bottom to top

    # Finding volumes adjacent to the bottom wall and associated ghost volumes
    curr = volNumb - 1
    ind = -1
    while curr > 0:

        wallVol[ind] = cv[curr,0]
        ghostWall[ind] = volArr[curr,4]
        left = volArr[curr,1]
        curr = left

        ind -= 1
    
    # Creating 2d array of volumes arrangement    
    s = []
    for j in range(len(x_cv)):
        # j = 0
        curr = int(wallVol[j])
        for k in range(len(cv)):
            x_cord = cv[curr,2]
            if cv[k,2] == x_cord:
                s.append(int(cv[k,0]))
            
        s.insert(0, s.pop())
        volSlice[:,j] = s
        s = []   
        
    # Setting the number of the volumes as integers in order to use them as indexes
    volSlice = volSlice.astype(int)
    
    # Calculating y-direction distanc ebetween CV centroids for integrations to come
    y_cv = u.copy()
    a = 0
    for b in range(len(y_cv)):
        # for a in range(len(volSlice[:,-1])):
        aa = volSlice[a,-1]
        volume = int(cv[aa,0])
        y_coord = cv[volume,3]
        y_cv[b] = y_coord            # y-coord of CV centroidS
        a += 1
   
    # Calculations
    for i in range(len(x_cv)):
        ### Calculating Average temperature at the current section           
        for j in range(1,len(y_cv)+1):
            foo = volSlice[-j,i]
            u[j-1] = data[foo,1]
            theta[j-1] = tData[foo,1]
            uTheta[j-1] = u[j-1]*theta[j-1]
            # print(foo)
              
            # print('{:.2E}'.format(uTheta[j]))
              
        Tm[i] = np.trapz(uTheta,np.flip(y_cv))/H
        # print(Tm[i])
        
        wv = volSlice[-2,i]    # number of the volume adjacent to the bottom wall (Index counting starting at top wall)
        gv = volSlice[-1,i]    # number of associated ghost volume
        wwv = volArr[wv,2]     # volume above wv (to get derivatives between them)
        # w = int(wallVol[i])
        
        
        thetaWall[i] = (tData[wv,1] + tData[gv,1])/2
        theta1[i] = (tData[wv,1] + tData[wwv,1])/2
        nu_x[i] = (2*(theta1[i] - thetaWall[i])/(cv[wv,3] + cv[wwv,3]))/(Tm[i] - thetaWall[i])   # theta derivative between y coord of wall volume and 0 (wall y-coord)
        
        # print('{}\t {}'.format(wv,gv))
        
    nu_avg = np.trapz(nu_x,x_cv)/Ls#((len(x_cv) - 1)*(cv[(len(volArr) - 2),2] - cv[(len(volArr) - 3),2]))
    
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('thetaM(x)')
    plt.plot(x_cv,Tm)
    plt.figure(2)
    plt.xlabel('x')
    plt.ylabel('Nu(x)')
    plt.plot(x_cv,nu_x)
    
    return nu_avg,nu_x,Tm,x_cv#,y_cv,theta1,thetaWall