#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:26:00 2021

@author: alegretti
"""
import numpy as np
from numba import jit

def makeSlices(nx1,nx3,ny1,ny3,volArr,cv,volNumb):
    
    yEnt = np.zeros((ny1-1))                        # y coord. of volumes centroids
    ySud = np.zeros((ny1+ny3-1))                    # y coord. of volumes centroids
    xAll = np.zeros((nx1+nx3-1))                      # x coord. of volumes centroids
    
    # Matching volumes at the bottom wall with its respective ghosts volumes
    wallVols = np.zeros((nx1+nx3-1,2))              # wallVols = [volWall,Ghost]
    
    count = 0
    for i in range(volNumb):
        if volArr[i,-1] < 0:
            wallVols[count,0] = int(i)
            wallVols[count,1] = int(volArr[i,-1])
            count+=1
    
    # Assembling 2D representation of volumes arrangement in entrance region and mapping y_cv coord.
    sliceEnt = np.zeros((ny1-1,nx1))
    for k in range(nx1-1):
        bw = int(wallVols[k,0])
        j = 0
        for i in range(volNumb):
            if cv[i,2] == cv[bw,2]:
                # print(i)
                sliceEnt[j,k] = int(i)
                yEnt[j] = cv[i,3]
                j += 1     
    
    # Assembling 2D representation of volumes arrangement in expanded region            
    sliceSud = np.zeros((ny1+ny3-1,nx3-1))
    for k in range(nx3-1):
        bw = int(wallVols[k+nx1,0])
        j = 0
        for i in range(volNumb):
            if cv[i,2] == cv[bw,2]:
                # print(i)
                sliceSud[j,k] = int(i)
                ySud[j] = cv[i,3]
                j += 1

    k = 0
    for i in range(volNumb):
        if volArr[i,-1] < 0:
            xAll[k] = cv[i,2]
            k += 1
        

    return wallVols,sliceEnt,sliceSud,yEnt,ySud,xAll            
                

def calcNu2(nx1,nx3,ny1,ny3,volArr,cv,volNumb,edge,h_data,t_data,Le,Ls):
    
    wallVols,sliceEnt,sliceSud,yEnt,ySud,xAll = makeSlices(nx1,nx3,ny1,ny3,volArr,cv,volNumb)
    
    uThetaEnt = sliceEnt.copy()             # products between u and T
    uThetaSud = sliceSud.copy()             # products between u and T
    
    uEnt = uThetaEnt.copy()
    for j in range(len(sliceEnt)):
        for i in range(len(sliceEnt[0])):
            a = int(sliceEnt[j,i])
            uEnt[j,i] = h_data[a,1]
    
    uSud = uThetaSud.copy()
    for j in range(len(sliceSud)):
        for i in range(len(sliceSud[0])):
            a = int(sliceSud[j,i])
            uSud[j,i] = h_data[a,1]    
    
    for j in range(len(sliceEnt)):
        for i in range(len(sliceEnt[0])):
            a = int(sliceEnt[j,i])
            uThetaEnt[j,i] = h_data[a,1]*t_data[a,1]
    
    for j in range(len(sliceSud)):
        for i in range(len(sliceSud[0])):
            a = int(sliceSud[j,i])
            uThetaSud[j,i] = h_data[a,1]*t_data[a,1]
    
    Tm = np.zeros((nx1+nx3-1))
    Nu_x = np.zeros((nx1+nx3-1))
    
    # Calculating Tm for entrance region
    for i in range(edge-int(wallVols[0,0])-1):
        U = 1.0     # average velocity
        h = 0.5     # channel height
        Tm[i] = (1/(U*h))*np.trapz(uThetaEnt[:,i],-yEnt)
        
    # Calculating Tm for expanded region
    skip = int(nx1)#len(range(edge-int(wallVols[0,0])))
    for i in range(nx1,nx3+nx1-1):
        U = 0.5     # average velocity
        h = 1.0     # channel height
        Tm[i] = (1/(U*h))*np.trapz(uThetaSud[:,i-skip],-ySud)        
        
    # Calculating Nu_x
    for i in range(len(Nu_x)):
        vol = int(wallVols[i,0])
        ghost = int(wallVols[i,1])
        Nu_x[i] = - (1/(1 - Tm[i]))*(t_data[vol,1] - t_data[ghost,1])/(cv[vol,3] - cv[ghost,3])
        
    Nu_AVG = (1/(Le+Ls))*np.trapz(np.sqrt(Nu_x**2),xAll)
    
    
    return Tm,xAll,Nu_x,Nu_AVG

def makeSlices_PP(nx1,nx3,ny1,ny3,volArr,cv,volNumb):
    
    yEnt = np.zeros((ny1-1))                      # y coord. of volumes centroids
    xAll = np.zeros((nx1-1))                      # x coord. of volumes centroids
    
    # Matching volumes at the top wall with its respective ghosts volumes
    wallVols = np.zeros((nx1-1,2))              # wallVols = [volWall,Ghost]
    
    count = 0
    for i in range(volNumb):
        if volArr[i,2] < 0: #TOP
            wallVols[count,0] = int(i)
            wallVols[count,1] = int(volArr[i,2])
            count+=1
    
    # Assembling 2D representation of volumes arrangement and mapping y_cv coord.
    sliceEnt = np.zeros((ny1-1,nx1-1))
    for k in range(nx1-1):
        bw = int(wallVols[k,0])
        j = 0
        for i in range(volNumb):
            if cv[i,2] == cv[bw,2]:
                # print(i)
                sliceEnt[j,k] = int(i)
                yEnt[j] = cv[i,3]
                j += 1     
    
    k = 0
    for i in range(volNumb):
        if volArr[i,-1] < 0:
            xAll[k] = cv[i,2]
            k += 1
        

    return wallVols,sliceEnt,yEnt,xAll       

def calcNu2_PP(nx1,ny1,volArr,cv,volNumb,h_data,t_data,Le,h):
    
    wallVols,sliceEnt,yEnt,xAll = makeSlices_PP(nx1,0,ny1,0,volArr,cv,volNumb)
    
    uThetaEnt = sliceEnt.copy()             # products between u and T
    
    uEnt = uThetaEnt.copy()
    for j in range(len(sliceEnt)):
        for i in range(len(sliceEnt[0])):
            a = int(sliceEnt[j,i])
            uEnt[j,i] = h_data[a,1]
    
    for j in range(len(sliceEnt)):
        for i in range(len(sliceEnt[0])):
            a = int(sliceEnt[j,i])
            uThetaEnt[j,i] = h_data[a,1]*t_data[a,1]

    Tm = np.zeros((nx1-1))
    Nu_x = np.zeros((nx1-1))

    # Calculating Tm 
    for i in range(len(sliceEnt[0])):
        U = 1.0     # average velocity
        Tm[i] = (1/(U*h))*np.trapz(uThetaEnt[:,i],-yEnt)        

    # Calculating Nu_x
    for i in range(len(Nu_x)):
        vol = int(wallVols[i,0])
        ghost = int(wallVols[i,1])
        Nu_x[i] = - (t_data[vol,1] - t_data[ghost,1])/(cv[vol,3] - cv[ghost,3])
        # print(Nu_x[i])

    Nu_AVG = (1/(Le))*np.trapz(np.sqrt(Nu_x**2),xAll)
    
    return Tm,xAll,Nu_x,Nu_AVG,uEnt,uThetaEnt,sliceEnt

@jit(nopython=True)
def calcNu2_LDC(nxy,volArr,cv,volNumb,h_data,t_data,Le,h,face_index,intDir,normalDir):
    """
    face_index = [1] east; [2] north; [3] west; [4] south
    intDir = itegration direction [4] x-dir/ [5] y-dir
    normalDir = normal direction [2] x-dir/ [3] y-dir
    """
    
    wallVols = np.zeros((nxy-1,2))              # wallVols = [volWall,Ghost]
    
    count = 0
    
    for i in range(volNumb):
        if volArr[i,face_index] < 0: # Boundary
            wallVols[count,0] = int(i)
            wallVols[count,1] = int(volArr[i,face_index])
            count+=1
    
    Nu_AVG = 0.0
    for i in range(nxy-1):
        vol = int(wallVols[i,0])
        ghost = int(wallVols[i,1])
        
        Nu_AVG += np.abs(((t_data[vol,1] - t_data[ghost,1])/(cv[vol,normalDir] - cv[ghost,normalDir]))*cv[i,intDir])
    
    return Nu_AVG