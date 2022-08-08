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

def LDC_bottomVort(volnumb,cv,volarr,h_data):
    
    """ Called inside next function"""
    
    # Interpolates values at nodes location and exports profiles
    #
    n = int(np.sqrt(volnumb))
    profile_bottom = np.zeros((n,2))
    
    k = 0    
    for i in range(volnumb):
        S = volarr[i,4]
        if S < 0:
            profile_bottom[k,0] = cv[i,2]
            profile_bottom[k,1] = (h_data[i,4] + h_data[S,4])/2  # vort
            
            k += 1
    
    myFileProf =  open("profile_vort_bottom_AVG={:.4f}.dat".format(np.mean(profile_bottom[:,1])), 'w')
    myFileProf.write('Variables = "x","w"\n')
    myFileProf.write('\n')
    for i in range(len(profile_bottom)):
        myFileProf.write('{}\t {}\n'.format(profile_bottom[i,0],profile_bottom[i,1]))
    
    myFileProf.close()


def LDC_profiles(cv,h_data,m_data,t_data,volarr,volnumb,xpos=0.5,ypos=0.5):
    
    # Interpolates values at nodes location and exports profiles
    #
    n = int(np.sqrt(volnumb)) + 1
    profile = np.zeros((n,9))
    
    k = 1
    # Interior points
    for i in range(volnumb):
        if cv[i,2] == xpos:
            N = volarr[i,2]
            profile[n-k,0] = (cv[i,3] + cv[N,3])/2          # y coord
            profile[n-k,1] = (h_data[i,1] + h_data[N,1])/2  # u vel
            profile[n-k,2] = (h_data[i,2] + h_data[N,2])/2  # v vel
            profile[n-k,3] = (h_data[i,3] + h_data[N,3])/2  # psi
            profile[n-k,4] = (h_data[i,4] + h_data[N,4])/2  # vort
            profile[n-k,5] = (m_data[i,-2] + m_data[N,-2])/2  # Mx
            profile[n-k,6] = (m_data[i,-1] + m_data[N,-1])/2  # My
            profile[n-k,7] = (t_data[i,1] + t_data[N,1])/2  # theta
            profile[n-k,8] = (m_data[i,4]*m_data[i,5] - m_data[i,3]*m_data[i,6] + m_data[N,4]*m_data[N,5] - m_data[N,3]*m_data[N,6])/2  # MxH
            k += 1
            S = volarr[i,4]
            # Bottom wall and south ghosts  
            if cv[i,2] == xpos and S < 0:
                profile[0,0] = (cv[i,3] + cv[S,3])/2          # y coord
                profile[0,1] = (h_data[i,1] + h_data[S,1])/2  # u vel
                profile[0,2] = (h_data[i,2] + h_data[S,2])/2  # v vel
                profile[0,3] = (h_data[i,3] + h_data[S,3])/2  # psi
                profile[0,4] = (h_data[i,4] + h_data[S,4])/2  # vort
                profile[0,5] = (m_data[i,-2] + m_data[S,-2])/2  # Mx
                profile[0,6] = (m_data[i,-1] + m_data[S,-1])/2  # My
                profile[0,7] = (t_data[i,1] + t_data[S,1])/2  # theta
                profile[0,8] = (m_data[i,4]*m_data[i,5] - m_data[i,3]*m_data[i,6] + m_data[S,4]*m_data[S,5] - m_data[S,3]*m_data[S,6])/2  # MxH
            
    myFileProf =  open("profileVert_x={:.2f}.dat".format(xpos), 'w')
    myFileProf.write('Variables = "y","u","v","psi","w","Mx","My","theta","MxH" \n')
    myFileProf.write('\n')
    for i in range(len(profile)):
        myFileProf.write('{}\t {}\t {}\t {}\t {}\t {}\t {}\t {}\t {}\n'.format(profile[i,0],profile[i,1],profile[i,2],profile[i,3],profile[i,4],profile[i,5],profile[i,6],profile[i,7],profile[i,8]))
    
    myFileProf.close()
    
    k = 1
    # Interior points
    for i in range(volnumb):
        if cv[i,3] == ypos:
            W = volarr[i,1]
            if cv[i,3] == ypos and W < 0:
                profile[0,0] = (cv[i,2] + cv[W,2])/2          # x coord
                profile[0,1] = (h_data[i,1] + h_data[W,1])/2  # u vel
                profile[0,2] = (h_data[i,2] + h_data[W,2])/2  # v vel
                profile[0,3] = (h_data[i,3] + h_data[W,3])/2  # psi
                profile[0,4] = (h_data[i,4] + h_data[W,4])/2  # vort
                profile[0,5] = (m_data[i,-2] + m_data[W,-2])/2  # Mx
                profile[0,6] = (m_data[i,-1] + m_data[W,-1])/2  # My
                profile[0,7] = (t_data[i,1] + t_data[W,1])/2  # theta
                profile[0,8] = (m_data[i,4]*m_data[i,5] - m_data[i,3]*m_data[i,6] + m_data[W,4]*m_data[W,5] - m_data[W,3]*m_data[W,6])/2  # MxH
            
            E = volarr[i,3]
            profile[k,0] = (cv[i,2] + cv[E,2])/2          # x coord
            profile[k,1] = (h_data[i,1] + h_data[E,1])/2  # u vel
            profile[k,2] = (h_data[i,2] + h_data[E,2])/2  # v vel
            profile[k,3] = (h_data[i,3] + h_data[E,3])/2  # psi
            profile[k,4] = (h_data[i,4] + h_data[E,4])/2  # vort
            profile[k,5] = (m_data[i,-2] + m_data[E,-2])/2  # Mx
            profile[k,6] = (m_data[i,-1] + m_data[E,-1])/2  # My
            profile[k,7] = (t_data[i,1] + t_data[E,1])/2  # theta
            profile[k,8] = (m_data[i,4]*m_data[i,5] - m_data[i,3]*m_data[i,6] + m_data[E,4]*m_data[E,5] - m_data[E,3]*m_data[E,6])/2  # MxH
            k += 1
            # left wall
        
    myFileProf =  open("profileHoriz_y={:.2f}.dat".format(ypos), 'w')
    myFileProf.write('Variables = "x","u","v","psi","w","Mx","My","theta" \n')
    myFileProf.write('\n')
    for i in range(len(profile)):
        myFileProf.write('{}\t {}\t {}\t {}\t {}\t {}\t {}\t {}\n'.format(profile[i,0],profile[i,1],profile[i,2],profile[i,3],profile[i,4],profile[i,5],profile[i,6],profile[i,7]))
    
    myFileProf.close()    
    
    LDC_bottomVort(volnumb,cv,volarr,h_data)
    
    return 