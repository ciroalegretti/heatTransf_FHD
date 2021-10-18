#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:07:43 2021

@author: alegretti
"""
from pre import loadGrid,defTimeStep,startArrays,clegg
from numerics.magnetization import equilibriumMag,solveMag
from numerics.vorticity import bc_vort,macCormack_w,vort_cg
from numerics.energy import bc_theta,macCormack_theta
from numerics.poisson import bc_uvPsi,eulerExp,uAndV,poisson_cg
from post import calcXr,calcNu,dataOut,endPrint
import numpy as np
import time

t0 = time.time()

""" Grids avaliable: 12383,19379,28173,38329,50363,63679,78953,95429,113943 """
volNumb = 12383# input('Enter volume count: ' )
Le = 5.0 #input('Enter entrance domain lenght: ')
Ls = 15.0 #input('Enter expansion domain lenght: ')
ER = 2.0 #input('Enter expansion ratio')
beta = 3.0 # grid streching parameter

<<<<<<< HEAD
<<<<<<< HEAD
folderTag = 'testeServ'

dtFac = 0.05
tol = 1E-3
=======
folderTag = 'fullCG'
=======
folderTag = 'vortCG'
>>>>>>> fbb97708d207a1728e37e79f9f28d96aa88eb4ac

dtFac = 1.0
tol = 1E-6
>>>>>>> 884042b990f7f22dae896d69424186c8b4b4809a
#phiArr = np.linspace(0.025,0.25,10)
#phiArr = np.insert(phiArr,0,1E-5)
#PeArr = np.logspace(-2,2,5)
# tolArr = np.logspace(-3,-8,6)
# PrArray = np.logspace(-1,1,5)
# alphaArr = np.logspace(-1,1,5)
# vols = np.array((12383,19379,28173,38329,50363,63679,78953,95429,113943))
# ReArr = np.linspace(50,400,8) #np.array((200.,400))
betaArr = np.array((3.0,))#2.0,3.0,4.0))

M_d = 1.0
Hx = 0.0
Hy = 1.0

alpha0 = 1.0
phi = 1E-5
Re = 50.0
Pe = 1.0
Pr = 0.7
Ec = 1E-8
beta_dt = 0.1  # beta_dt = delta T/T_C

# for volNumb in vols:
# for Re in ReArr:
for beta in betaArr:

    nx1,nx2,nx3,ny1,ny2,ny3,volArr,volNodes,nodesCoord,x1,x2,x3,y1,y2,y3,e,edgeVol,CVdata,volNumb,nodeNumb,Le,Ls,S,wallFunc = loadGrid.loadGrid(volNumb,Le,Ls,ER,beta)
    
    h_data,m_data,t_data,flux = startArrays.starters1d(volArr,CVdata,Hx,Hy,alpha0)
    
    dt = defTimeStep.dtMin(Re,Pr,dtFac,x1,x3,y2,y3)
    
    # m_data,H0 = clegg.clegg(m_data,volNumb,CVdata,S,Le)
    
    m_data,t_data = equilibriumMag.equilibriumMag(m_data,t_data,phi,M_d,alpha0,beta_dt,nx1,ny1,volNumb,CVdata,Le)  # Calculating equilibrium magnetization
    m_data[:,5] = m_data[:,1]                                      # Setting initial x-magnetization as M0
    m_data[:,6] = m_data[:,2]                                      # Setting initial y-magnetization as M0
    
        
    h_data = bc_uvPsi.bc_uvPsi(h_data,volArr,CVdata,edgeVol,S)
    # m_data = main_func.bc_1d_mag(m_data,volArr,CVdata,edgeVol,M0x,M0y,Le)
    t_data = bc_theta.bc_theta(t_data,volArr,CVdata,edgeVol,S)
    i = 0
    erroPsi,erroW,erroMx,erroMy,erroTheta = 1.0,1.0,1.0,1.0,1.0
    
    while max(erroW,erroPsi,erroMx,erroMy,erroTheta) > tol:
    
        """Copying values before iteration"""
        dataDummy = h_data.copy()   
        m_dataDummy = m_data.copy()  
        t_dataDummy = t_data.copy()
        """Enforcing BC for stream function and velocity field"""
        h_data = bc_uvPsi.bc_uvPsi(h_data,volArr,CVdata,edgeVol,S)
        """Solving Poisson equation for streamfunction"""
        # h_data = poisson_cg.poisson_cg(volNumb,volArr,CVdata,h_data)
        h_data = eulerExp.poissonExplicit(h_data,volArr,volNodes,CVdata)
        """Calculating velocity field components"""
        h_data = uAndV.calculateUandV(h_data,volArr,CVdata)
        """Enforcing BC for vorticity""" 
        h_data = bc_vort.bc_w(h_data,volArr,CVdata,edgeVol,S)
        """Solving vorticity equation"""
        h_data = vort_cg.vort_cg(volNumb,volArr,CVdata,h_data,dt,Re)
        # h_data = macCormack_w.macCormack_w(h_data,m_data,Re,flux,wallFunc,volArr,CVdata,dt,Pe,phi)
        """Enforcing temperature BC"""
        # t_data = bc_theta.bc_theta(t_data,volArr,CVdata,edgeVol,S)
        """Solving energy convection-difusion equation with magnetic source terms"""
        # t_data = macCormack_theta.macCormack_theta(m_data,h_data,t_data,Re*Pr,flux,wallFunc,volArr,CVdata,dt,phi,Ec,Re,Pe)
        """Updating alpha and equilibrium magnetization as temperature functions"""
        # m_data,t_data = equilibriumMag.equilibriumMag(m_data,t_data,phi,M_d,alpha0,beta_dt,nx1,ny1,volNumb,CVdata,Le)  # Calculating equilibrium magnetization
        """Solving magnetization equation"""
        # magData = solveMag.ftcsMx(h_data,m_data,volArr,CVdata,volNodes,nodesCoord,Re,dt,phi,Pe)
        # magData = solveMag.ftcsMy(h_data,m_data,volArr,CVdata,volNodes,nodesCoord,Re,dt,phi,Pe)
    
        erroTheta = np.max(np.abs(t_data[:,1] - t_dataDummy[:,1]))
        erroPsi = np.max(np.abs(h_data[:,3] - dataDummy[:,3]))
        erroW = np.max(np.abs(h_data[:,4] - dataDummy[:,4]))
        erroMx = np.max(np.abs(m_data[:,5] - m_dataDummy[:,5]))
        erroMy = np.max(np.abs(m_data[:,6] - m_dataDummy[:,6]))
    
        i+= 1
        # print(i)
        if i % 1000 == 0:
            print('{}k_it\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}'.format(int(i/1000),erroPsi,erroW,erroMx,erroMy,erroTheta))
       
    Xr = calcXr.calcXr(edgeVol,volArr,h_data,CVdata,Le)
    Nu_AVG,Nu_x,thetaM_x,x_cv = calcNu.calcNu(np.max(y2),nx3,ny2,ny3,x3,Le,h_data,t_data,CVdata,volArr,Ls,volNumb)
    
    dataOut.createFolderHydro(folderTag,ER,Re,Le,Ls,volNumb,tol,Xr,S,dtFac,phi,Pe,alpha0,Pr,Nu_AVG,beta)
    dataOut.exportData(CVdata,h_data,m_data,t_data,volNumb,nodeNumb,volNodes,nodesCoord,x_cv,thetaM_x,Nu_x,Nu_AVG)
    
    endPrint.endPrint(Re,Xr,S,Nu_AVG,t0)
