#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 09:25:39 2022

@author: alegretti
"""
import numpy as np

def profilesOut(xpos,arr2d,cv,h_data,t_data,m_data,volarr,volnumb):
    
    # Interpolates values at nodes location and exports profiles
    #
    # xpos must be coincident with a x coord of volume centroid 
    
    nodes = len(arr2d) + 1
    profile = np.zeros((nodes,8))
    
    k = 1
    # Interior points
    for i in range(volnumb):
        if cv[i,2] == xpos:
            N = volarr[i,2]
            profile[nodes-k,0] = (cv[i,3] + cv[N,3])/2          # y coord
            profile[nodes-k,1] = (h_data[i,1] + h_data[N,1])/2  # u vel
            profile[nodes-k,2] = (h_data[i,2] + h_data[N,2])/2  # v vel
            profile[nodes-k,3] = (h_data[i,3] + h_data[N,3])/2  # psi
            profile[nodes-k,4] = (h_data[i,4] + h_data[N,4])/2  # vort
            profile[nodes-k,5] = (m_data[i,-2] + m_data[N,-2])/2  # Mx
            profile[nodes-k,6] = (m_data[i,-1] + m_data[N,-1])/2  # My
            profile[nodes-k,7] = (t_data[i,1] + t_data[N,1])/2  # theta
            k += 1
       
    # Bottom wall and south ghosts        
    for i in range(volnumb):
        S = volarr[i,4]
        if cv[i,2] == xpos and S < 0:
            profile[0,0] = (cv[i,3] + cv[S,3])/2          # y coord
            profile[0,1] = (h_data[i,1] + h_data[S,1])/2  # u vel
            profile[0,2] = (h_data[i,2] + h_data[S,2])/2  # v vel
            profile[0,3] = (h_data[i,3] + h_data[S,3])/2  # psi
            profile[0,4] = (h_data[i,4] + h_data[S,4])/2  # vort
            profile[0,5] = (m_data[i,-2] + m_data[S,-2])/2  # Mx
            profile[0,6] = (m_data[i,-1] + m_data[S,-1])/2  # My
            profile[0,7] = (t_data[i,1] + t_data[S,1])/2  # theta
            
            
    myFileProf =  open("profile_x={}.dat".format(xpos), 'w')
    myFileProf.write('Variables = "y","u","v","psi","w","Mx","My","theta" \n')
    myFileProf.write('\n')
    for i in range(len(profile)):
        myFileProf.write('{}\t {}\t {}\t {}\t {}\t {}\t {}\t {}\n'.format(profile[i,0],profile[i,1],profile[i,2],profile[i,3],profile[i,4],profile[i,5],profile[i,6],profile[i,7]))
    
    myFileProf.close()
    
    return profile

def LDC_lidVort(nx1,re,arr2d,cv,h_data,volarr):
    
    # Interpolates values at nodes location and exports profiles
    #
    
    nodes = len(arr2d) + 1
    profile = np.zeros((nodes,2))
    
    k = 1
    # Top volumes
    for i in range(nx1+1):
        N = volarr[i,2]
        profile[nodes-k,0] = cv[i,2]           # x coord
        profile[nodes-k,1] = (h_data[i,-1] + h_data[N,-1])/2  # vort at interface
        k +=1
        
    myFileProf =  open("profileVortLid_Re={}.dat".format(re), 'w')
    myFileProf.write('Variables = "x","vort" \n')
    myFileProf.write('\n')
    for i in range(len(profile)):
        myFileProf.write('{}\t {}\t \n'.format(profile[i,0],profile[i,1]))
    
    myFileProf.close()
    
    return profile