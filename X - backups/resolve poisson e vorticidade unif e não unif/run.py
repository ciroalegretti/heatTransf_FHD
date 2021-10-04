#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:07:43 2021

@author: alegretti
"""
from misc import loadGrid, defTimeStep, idNeighbors, dataOut
from numerics import startArrays, macCormack
from numerics.magnetization import equilibriumMag
from numerics.vorticity import bc_vort,fluxVort
from numerics.energy import bc_theta
from numerics.poisson import bc_uvPsi, fluxPsi, uAndV
import numpy as np
import time

t0 = time.time()

volNumb = 12383# input('Enter volume count: ' )
Le = 5.0 #input('Enter entrance domain lenght: ')
Ls = 15.0 #input('Enter expansion domain lenght: ')
ER = 2.0 #input('Enter expansion ratio')
beta = 4.0

folderTag = 'foo'

dtFac = 0.05
tol = 1E-5

#phiArr = np.linspace(0.025,0.25,10)
#phiArr = np.insert(phiArr,0,1E-5)
#PeArr = np.logspace(-2,2,5)
# EcArr = np.linspace(0,10,11)
# tolArr = np.logspace(-3,-8,6)
# PrArray = np.logspace(-1,1,5)
# alphaArr = np.logspace(-1,1,5)
# vols = np.array((14778,))#,33506,60153,94091,136128,185276))
# ReArr = np.linspace(50,400,8) #np.array((200.,400))
betaArr = np.array((4.0))

M_d = 1.0
Hx = 0.0
Hy = 1.0

alpha = 1E-5
phi = 0.05#1E-5
Re = 1E-5
Pe = 1E+5
Pr = 0.7
Ec = 1E-8

"""############################################################################################################################################################"""

# for beta in betaArr:
    
nx1,nx2,nx3,ny1,ny2,ny3,volArr,volNodes,nodesCoord,x1,x2,x3,y1,y2,y3,e,edgeVol,CVdata,volNumb,nodeNumb,Le,Ls,S,wallFunc = loadGrid.loadGrid(volNumb,Le,Ls,ER,beta)

h_data,m_data,t_data,flux_xi,flux_mx,flux_my,flux_t,flux_psi,s_xi,s_mx,s_my,s_t,s_psi = startArrays.starters1d(volArr,CVdata)

dt = defTimeStep.dtMin(Re,Pr,dtFac,x1,x3,y2,y3)

m_data = equilibriumMag.equilibriumMag(m_data,phi,M_d,alpha,Hx,Hy,nx1,ny1,volNumb,CVdata,Le)  # Calculating equilibrium magnetization
m_data[:,5] = m_data[:,1]                                      # Setting initial x-magnetization as M0
m_data[:,6] = m_data[:,2]                                      # Setting initial y-magnetization as M0
    
h_data = bc_uvPsi.bc_flow(h_data,volArr,CVdata,edgeVol,S)
# m_data = main_func.bc_1d_mag(m_data,volArr,CVdata,edgeVol,M0x,M0y,Le)
t_data = bc_theta.bc_theta(t_data,volArr,CVdata,edgeVol,S)
i = 0
erroPsi,erroW,erroMx,erroMy,erroTheta = 1.0,1.0,1.0,1.0,1.0

    
while max(erroW,erroPsi,erroMx,erroMy,erroTheta) > tol:
# while i <100:
    """Copying values before iteration"""
    dataDummy = h_data.copy()   
    m_dataDummy = m_data.copy()  
    t_dataaDummy = t_data.copy()
    """Enforcing BC for stream function and velocity field"""
    h_data = bc_uvPsi.bc_flow(h_data,volArr,CVdata,edgeVol,S)
    """Solving Poisson equation for streamfunction"""
    h_data = macCormack.poissonExplicit(h_data,volArr,volNodes,CVdata)
    # h_data,flux_psi = macCormack.macCormack_psi(h_data,flux_psi,volArr,CVdata,dt)
    """Calculating velocity field components"""
    h_data = uAndV.calculateUandV(h_data,volArr,CVdata)
    """Enforcing BC for vorticity""" 
    h_data = bc_vort.bc_w(h_data,volArr,CVdata,edgeVol,S)
    """Solving vorticity equation"""
    h_data = macCormack.macCormack_XI(h_data,Re,flux_xi,s_xi,wallFunc,volArr,CVdata,dt)
    
    erroTheta = np.max(np.abs(t_data[:,1] - t_dataaDummy[:,1]))
    erroPsi = np.max(np.abs(h_data[:,3] - dataDummy[:,3]))
    erroW = np.max(np.abs(h_data[:,4] - dataDummy[:,4]))
    erroMx = np.max(np.abs(m_data[:,5] - m_dataDummy[:,5]))
    erroMy = np.max(np.abs(m_data[:,6] - m_dataDummy[:,6]))
    
    
    i+= 1
    # if i % 1000 == 0:
    print('{}\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}'.format(i,erroPsi,erroW,erroMx,erroMy,erroTheta))


h_data = uAndV.calculateUandV(h_data,volArr,CVdata)
Xr = 0.0
Nu_AVG = 0.0

dataOut.createFolderHydro(folderTag,ER,Re,Le,Ls,volNumb,tol,Xr,S,dtFac,phi,Pe,alpha,Pr,Ec,Nu_AVG,beta)
dataOut.exportData(CVdata,h_data,m_data,t_data,volNumb,nodeNumb,volNodes,nodesCoord)#,x_cv,thetaM_x,Nu_x,Nu_AVG)














