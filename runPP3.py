#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runLDC.py

Steady-state solutions of the Graetz probLm under vorticity-stream
function formulation with magnetic effects and heat transfer

Created on Wed Jul 28 19:07:43 2021

@author: aLgretti
"""
from pre import loadGrid,defTimeStep,startArrays,externalField
from numerics.magnetization import equilibriumMag,equilibriumMagIvanov,solveMag
from numerics.vorticity import bc_vort,macCormack_w,vort_cgs
from numerics.energy import bc_theta,macCormack_theta,energy_cgs
from numerics.poisson import bc_uvPsi,uAndV,poisson_cg,eulerExp
from post import calcNu,dataOut,profilesOut
import numpy as np
import os

# =============================================================================
h = 0.5                 # Channel height (so Dh = 1)
L = 20.0*h              # Channel width
nx = 201#[51,101,201,401]#1001
ny = 31
dtFac = 5.0             # Time step coefficient
tol = 1E-6              # Numerical tolerance for the outer iteration
volNumb = 6000
Pr = 100.               # Prandtl number
Ec = 1E-3               # Eckert number 
beta_dt = 0.1           # beta_dt = delta T/T_C
# =============================================================================
#               DEFINE VARIABLE PARAMETERS AND MAGNET GEOMETRY 
# =============================================================================
phiArr = np.linspace(0.01,0.25,20)
phiArr = np.insert(phiArr,0,1E-5)
PeArr = [1E-2]#[0.01, 0.1, 1.0]#np.linspace(0.01,10,4)
# PrArray = np.logspace(-1,1,5)
alphaArr = [2.7231] #np.linspace(0.1, 10.0, 20)
#alphaArr = np.insert(alphaArr,0,1E-5)
# alphaArr = np.insert(alphaArr,0,1E-5)
# alphaArr = alphaArr[15:]
# volsArr = [6000]#np.linspace(5000,40000,8)#[2500,5000,10000,20000]#[10000,20000,50000]
invGzArr = [.1]#np.logspace(1,-5,7)#[0.01]#,1E-5]#             # Inverse of Graetz number [1.0, 0.1, 0.01]
# tolArr = [1E-6]#np.logspace(-4,-10,7)

lambArr = [10.2118] #np.linspace(1,3,3)               # Lambda parameter for chain formation (Ivanov - M0)

outTag = 'a=10nm - ivanov'

dist = 0                # Vertical distance between the magnet and the bottom wall
ratio = 0.25               # (channel lenght)/(magnet face lenght)

profilesLoc = [2.025,4.025,5.025,8.025]#[2.525, 5.025, 7.525]
# =====================================================================
#            Define external field H and start arrays
# =====================================================================
# magnet = 'horiz'
# magnet = 'vert' 
magnetArr = ['vert']#['horiz','vert']

# fieldSty = 'clegg'
#fieldSty = 'gradUnif'
# fieldSty = 'unif'
fieldStyArr = ['clegg']#['clegg','gradUnif','unif']
# =============================================================================
# 
#                               MAIN LOOP 
# 
# =============================================================================
for invGz in invGzArr:
    for Pe in PeArr:
        for alpha0 in alphaArr:
            for phi in phiArr:
                for magnet in magnetArr:
                    for fieldSty in fieldStyArr:
                        for lamb in lambArr:
                            
                            Re = L/(2*h*invGz*Pr)
                            # folderTag = 'invGz={:.1E}'.format(invGz)
        
                            # Grid Import    
                            volArr,volNodes,nodesCoord,x1,y1,CVdata,volNumb,nodeNumb,\
                                L,h,wallFunc = loadGrid.loadGrid_PP(L,h,volNumb,nx,ny)
                            # =====================================================================
                            eh = externalField.inputField(magnet)    
                            folderTag = '{}_{}_{}'.format(outTag,magnet,fieldSty)
                            x0,y0,b,Lm = externalField.getCoordPP(eh,L,h,dist,ratio)
                            dt = defTimeStep.dtMin(Re,Pr,dtFac,CVdata)
                            h_data,m_data,t_data,modH = startArrays.starters1d_PP(fieldSty,volArr\
                                                    ,CVdata,alpha0,h,L,eh,x0,y0,b,Lm,phi,beta_dt)
        
                            i,erroPsi,erroW,erroMx,erroMy,erroTheta = 0,1.0,1.0,1.0,1.0,1.0
                            # myfile = open('./Gz-1={}_R2_{}vols_tol{:.1E}.dat'.format(invGz,volNumb,tol),'w')
                            # myfile.write('Variables="it","resW","resPsi","resTheta" \n')
        
                            # =====================================================================
                            while max(erroW,erroPsi,erroMx,erroMy,erroTheta) > tol:
        
                                """Copying values before iteration"""
                                dataDummy = h_data.copy()
                                m_dataDummy = m_data.copy()
                                t_dataDummy = t_data.copy()
                                # if erroW > tol:
                                """Enforcing BC for stream function and velocity field"""
                                h_data = bc_uvPsi.bc_uvPsi_PP(h_data,volArr,CVdata,h)
                                """Solving Poisson equation for streamfunction"""
                                # h_data = poisson_cg.poisson_cgs(volNumb,volArr,CVdata,h_data,dt)
                                h_data = eulerExp.poissonExplicit(h_data,volArr,volNodes,CVdata)
                                """Calculating velocity field components"""
                                h_data = uAndV.calculateUandV(h_data,volArr,CVdata)
                                """Enforcing BC for vorticity"""
                                h_data = bc_vort.bc_w_PP(h_data,volArr,CVdata,h)
                                """Solving vorticity equation"""
                                h_data = vort_cgs.vort_cgs(volNumb,wallFunc,volArr,CVdata,h_data,\
                                                          m_data,dt,Re,Pe,phi)
                                """Enforcing temperature BC"""
                                t_data = bc_theta.bc_theta_PP(t_data,volArr,CVdata)
                                """Solving energy equation with magnetic source terms"""
                                t_data = energy_cgs.energy_cgs(volNumb,volArr,CVdata,wallFunc,\
                                                        h_data,t_data,m_data,phi,Ec,Re,Pe,Pr,dt)
                                # t_data = macCormack_theta.macCormack_theta(m_data,h_data,t_data,Re*Pr,wallFunc,volArr,CVdata,dt,phi,Ec,Re,Pe)
                                """Updating' alpha and equilibrium mag as temperature functions"""
                                #m_data,t_data = equilibriumMag.equilibriumMag(m_data,t_data,phi,\
                                      #                                        alpha0,beta_dt,volNumb) 
                                m_data,t_data = equilibriumMagIvanov.equilibriumMagIvanov(m_data,t_data,phi,\
                                                                               alpha0,beta_dt,volNumb,lamb)
                                """Solving magnetization equation"""
                                magData = solveMag.ftcsMx(h_data,m_data,volArr,CVdata,volNodes,\
                                                                           nodesCoord,dt,phi,Pe)
                                magData = solveMag.ftcsMy(h_data,m_data,volArr,CVdata,volNodes,\
                                                                           nodesCoord,dt,phi,Pe)
    
                                erroTheta = np.max(np.abs(t_data[:,1] - t_dataDummy[:,1]))
                                erroPsi = np.max(np.abs(h_data[:,3] - dataDummy[:,3]))
                                erroW = np.max(np.abs(h_data[:,4] - dataDummy[:,4]))
                                erroMx = np.max(np.abs(m_data[:,5] - m_dataDummy[:,5]))
                                erroMy = np.max(np.abs(m_data[:,6] - m_dataDummy[:,6]))
    
                                i+= 1
                                # print(i)
                                if i % 1000 == 0:
                                    print('{}k_it\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}'\
                                          .format(int(i/1000),erroPsi,erroW,erroMx,erroMy,erroTheta))
                           
                                # myfile.write('{} \t {:.5E}\t {:.5E}\t {:.5E} \n'.format(i,erroW,erroPsi,erroTheta))
                            
                            # myfile.close()
        
                            Tm,xAll,Nu_x,Nu_AVG,uEnt,uThetaEnt,arr2d = calcNu.calcNu2_PP(nx,ny,\
                                                    volArr,CVdata,volNumb,h_data,t_data,L,h)
        
                            dataOut.createFolderHydro_PP(folderTag,Re,L,h,volNumb,tol,dtFac,phi,\
                                                                        Pe,alpha0,Pr,Nu_AVG,i,Ec,lamb)
        
                            for xpos in profilesLoc:
                                profile = profilesOut.profilesOut(xpos,arr2d,CVdata,h_data,t_data,m_data,volArr,volNumb)
        
                            dataOut.exportDataPP(CVdata,h_data,m_data,t_data,volNumb,nodeNumb,\
                                                    volNodes,nodesCoord,modH,xAll,Tm,Nu_x,Nu_AVG)                                
                               
                            os.chdir('../../../') 