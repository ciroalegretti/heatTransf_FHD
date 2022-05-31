#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:07:43 2021

@author: alegretti
"""
from pre import loadGrid,defTimeStep,startArrays,initialFlow,externalField
from numerics.magnetization import equilibriumMag,solveMag
from numerics.vorticity import bc_vort,macCormack_w,vort_cgs
from numerics.energy import bc_theta,macCormack_theta,energy_cgs
from numerics.poisson import bc_uvPsi,eulerExp,uAndV,poisson_cg
from post import calcXr,calcNu,dataOut,endPrint
import numpy as np
import time
import os
import sys

t0 = time.time()

""" beta = 1.0: 1953,8303,19053,34203,53753"""

""" Grids avaliable: 12383,19379,28173,38329,50363,63679,78953,95429,113943 """

"""                   Selected volNumb and beta                             """
# def initialFlow(tol,h_data,label):

#     print("\n No solution available, starting hydrodynamic computations \n")

#     i = 0
#     erroPsi = 1.0
#     erroW = 1.0

#     myfile = open('./initial/BFS/convXr_{}vols_tol{:.1E}.dat'.format(volNumb,tol),'w')
#     myfile.write('Variables="it","Xr"\n')

#     while max(erroW,erroPsi) > tol:

#         """Copying values before iteration"""
#         dataDummy = h_data.copy()   
#         """Enforcing BC for stream function and velocity field"""
#         h_data = bc_uvPsi.bc_uvPsi(h_data,volArr,CVdata,edgeVol,S)
#         """Solving Poisson equation for streamfunction"""
#         # h_data = poisson_cg.poisson2_cgs(edgeVol,volNumb,volArr,CVdata,h_data,ER,h,dt)
#         h_data = poisson_cg.poisson_cgs(volNumb,volArr,CVdata,h_data,dt)
#         # h_data = eulerExp.poissonExplicit(h_data,volArr,volNodes,CVdata)
#         """Calculating velocity field components"""
#         h_data = uAndV.calculateUandV(h_data,volArr,CVdata)
#         """Enforcing BC for vorticity""" 
#         h_data = bc_vort.bc_w(h_data,volArr,CVdata,edgeVol,S)
#         """Solving vorticity equation"""
#         h_data = vort_cgs.vort_cgs(volNumb,wallFunc,volArr,CVdata,h_data,m_data,dt,Re,Pe,1E-10)
#         # h_data = macCormack_w.macCormack_w(h_data,m_data,Re,flux,wallFunc,volArr,CVdata,dt,Pe,phi)

#         erroPsi = np.max(np.abs(h_data[:,3] - dataDummy[:,3]))
#         erroW = np.max(np.abs(h_data[:,4] - dataDummy[:,4]))

#         XrS = calcXr.calcXr(edgeVol,volArr,h_data,CVdata,Le,y1,S,nx3)

#         myfile.write('{} \t {:.5E} \n'.format(i,XrS))

#         i+= 1
#         # print(XrS)
#         if i % 1000 == 0:
#             print('{}k_it\t {:.2E}\t {:.2E}\t {:.3f}'.format(int(i/1000),erroPsi,erroW,XrS))
    
#     myfile.close()
    
#     dataOut.exportHydro(beta,ER,Re,Le,Ls,volNumb,h_data,nodeNumb,nodesCoord,volNodes)
#     print("\n Hydrodynamic solution computed and stored \n")
    
#     return h_data

# =============================================================================
volNumb = 12383
beta = 2.0 # grid streching parameter
Le = 5.0 #input('Enter entrance domain lenght: ')
Ls = 15.0 #input('Enter expansion domain lenght: ')
# LsArr = [15.0, 20.0, 25.0, 30.0]
ER = 2.0 #input('Enter expansion ratio')
h = 0.5
dtFac = 5.0            # Time step coefficient
# tol = 1E-6              # Numerical tolerance for the outer iteration
Pr = 100.               # Prandtl number
Ec = 1E-6               # Eckert number 
beta_dt = 0.1           # beta_dt = delta T/T_C
# =============================================================================
#               DEFINE VARIABLE PARAMETERS AND MAGNET GEOMETRY 
# =============================================================================
phiArr = [1E-5] #np.linspace(0.05,0.25,5)
#phiArr = np.insert(phiArr,0,1E-5)
PeArr = [1E+5]#[0.01, 0.1, 1.0]#np.linspace(0.01,10,4)
# PrArray = np.logspace(-1,1,5)
alphaArr = [1E-5]#[0.1, 1.0, 10]
# alphaArr = np.insert(alphaArr,0,1E-5)
# alphaArr = alphaArr[15:]
volsArr = [12383]#[12383,19379,28173,38329,50363,63679,78953,95429,113943]
ReArr = [400.,5000]#[400.,1000,2000.,3200.,5000.]
tolArr =[1E-9,1E-10]# np.logspace(-4,-10,7)

dist = 0                # Vertical distance between the magnet and the bottom wall
ratio = 2               # (channel lenght)/(magnet face lenght)

profilesLoc = [1.025,3.025,5.025,7.025,9.025]#[2.525, 5.025, 7.525]

outTag = 'debugXr'
# =============================================================================
# 
#                               MAIN LOOP 
# 
# =============================================================================
for Re in ReArr:
    for Pe in PeArr:
        for alpha0 in alphaArr:
            for phi in phiArr:
                for volNumb in volsArr:
                    for tol in tolArr:
    
                        # gridDataCheck = os.path.exists("./grids/BFS/ER={}/beta={}/gridData_{}_volumes_Le={}_Ls={}_ER={}.npz".format(ER,beta,volNumb,Le,Ls,ER))
                        
                        # if gridDataCheck == True:
                        nx1,nx2,nx3,ny1,ny2,ny3,volArr,volNodes,nodesCoord,x1,x2,x3,y1,y2,y3,e,edgeVol,CVdata,volNumb,nodeNumb,Le,Ls,S,wallFunc = loadGrid.loadGrid_BFS(volNumb,Le,Ls,ER,beta)
                        # else:
                        #     print('\n no Grid available \n')
                        #     sys.exit()
                        
                        # =====================================================================
                        #            Define external field H and start arrays
                        # =====================================================================
                        #magnet = 'horiz'
                        magnet = 'vert'
                        # fieldSty = 'clegg'
                        fieldSty = 'gradUnif'
                        # fieldSty = 'unif'
                        # =====================================================================
                        eh = externalField.inputField(magnet)    
                        folderTag = '{}_{}_{}'.format(outTag,magnet,fieldSty)
                        x0,y0,b,Lm = externalField.getCoordBFS(eh,Le,h,dist,ratio)
                        dt = defTimeStep.dtMin(Re,Pr,dtFac,CVdata)
                        h_data,m_data,t_data,modH = startArrays.starters1d_BFS(fieldSty,volArr\
                                                                    ,CVdata,alpha0,h,Ls,eh,x0,y0,b,Lm,phi,beta_dt)
                        
                        m_data,t_data = equilibriumMag.equilibriumMag(m_data,t_data,phi,\
                                                                                            alpha0,beta_dt,volNumb)   # Calculating equilibrium magnetization
                        i,erroPsi,erroW,erroMx,erroMy,erroTheta = 0,1.0,1.0,1.0,1.0,1.0
                        
                        # hydroResCheck = os.path.exists("./initial/BFS/beta={}_ER={}_Re={:.1E}_Le={}_Ls={}_Volumes={}".format(beta,ER,Re,Le,Ls,volNumb))
                        
                        myfile = open('./Re={}_R2_{}vols_tol{:.1E}.dat'.format(Re,volNumb,tol),'w')
                        myfile.write('Variables="it","resW","resPsi","resTheta" \n')
                        
                        
                        # if hydroResCheck == False:
                        #     h_data = initialFlow.initialFlow(tol,h_data,folderTag)
                        # else:
                        #     h_data = np.load("./initial/BFS/beta={}_ER={}_Re={:.1E}_Le={}_Ls={}_Volumes={}/hydroSol.npz".format(beta,ER,Re,Le,Ls,volNumb))['h_data']
                        #     print('\n Available hydrodynamic solution imported, starting coupled solution \n')
                        
                        while max(erroW,erroPsi,erroMx,erroMy,erroTheta) > tol:
                        
                            """Copying values before iteration"""
                            dataDummy = h_data.copy()   
                            m_dataDummy = m_data.copy()  
                            t_dataDummy = t_data.copy()
                            """Enforcing BC for stream function and velocity field"""
                            h_data = bc_uvPsi.bc_uvPsi(h_data,volArr,CVdata,edgeVol,S)
                            """Solving Poisson equation for streamfunction"""
                            # h_data = poisson_cg.poisson2_cgs(edgeVol,volNumb,volArr,CVdata,h_data,ER,h,dt)
                            # h_data = poisson_cg.poisson_cgs(volNumb,volArr,CVdata,h_data,dt)
                            h_data = eulerExp.poissonExplicit(h_data,volArr,volNodes,CVdata)
                            """Calculating velocity field components"""
                            h_data = uAndV.calculateUandV(h_data,volArr,CVdata)
                            """Enforcing BC for vorticity""" 
                            h_data = bc_vort.bc_w(h_data,volArr,CVdata,edgeVol,S)
                            """Solving vorticity equation"""
                            h_data = vort_cgs.vort_cgs(volNumb,wallFunc,volArr,CVdata,h_data,m_data,dt,Re,Pe,phi)
                            # h_data = macCormack_w.macCormack_w(h_data,m_data,Re,wallFunc,volArr,CVdata,dt,Pe,phi)
                            """Enforcing temperature BC"""
                            # t_data = bc_theta.bc_thetaBFS(t_data,volArr,CVdata,edgeVol,S)
                            """Solving energy convection-difusion equation with magnetic source terms"""
                            # t_data = macCormack_theta.macCormack_theta(m_data,h_data,t_data,Re*Pr,flux,wallFunc,volArr,CVdata,dt,phi,Ec,Re,Pe)
                            # t_data = energy_cgs.energy_cgs(volNumb,volArr,CVdata,wallFunc,h_data,t_data,m_data,phi,Ec,Re,Pe,Pr,dt)
                            """Updating alpha and equilibrium magnetization as temperature functions"""
                            # m_data,t_data = equilibriumMag.equilibriumMag(m_data,t_data,phi,\
                            #                                       alpha0,beta_dt,volNumb)   # Calculating equilibrium magnetization
                            # """Solving magnetization equation"""
                            # magData = solveMag.ftcsMx(h_data,m_data,volArr,CVdata,volNodes,nodesCoord,dt,phi,Pe)
                            # magData = solveMag.ftcsMy(h_data,m_data,volArr,CVdata,volNodes,nodesCoord,dt,phi,Pe)
                            
                            erroTheta = np.max(np.abs(t_data[:,1] - t_dataDummy[:,1]))
                            erroPsi = np.max(np.abs(h_data[:,3] - dataDummy[:,3]))
                            erroW = np.max(np.abs(h_data[:,4] - dataDummy[:,4]))
                            erroMx = np.max(np.abs(m_data[:,5] - m_dataDummy[:,5]))
                            erroMy = np.max(np.abs(m_data[:,6] - m_dataDummy[:,6]))
                        
                            i+= 1
                            # print(i)
                            if i % 1000 == 0:
                                print('{}k_it\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}'.format(int(i/1000),erroPsi,erroW,erroMx,erroMy,erroTheta))
                
                            myfile.write('{} \t {:.5E}\t {:.5E}\t {:.5E} \n'.format(i,erroW,erroPsi,erroTheta))
                            
                        myfile.close()            
                
                        XrS = calcXr.calcXr(edgeVol,volArr,h_data,CVdata,Le,y1,S,nx3)
                        Tm,xAll,Nu_x,Nu_AVG = calcNu.calcNu2(nx1,nx3,ny1,ny3,volArr,CVdata,volNumb,edgeVol,h_data,t_data,Le,Ls)
                        
                        dataOut.createFolderHydro(folderTag,ER,Re,Le,Ls,volNumb,tol,XrS[0],dtFac,phi,Pe,alpha0,Pr,Nu_x[-1],beta,i)
                        dataOut.exportData(CVdata,h_data,m_data,t_data,volNumb,nodeNumb,volNodes,nodesCoord,modH,xAll,Tm,Nu_x,Nu_AVG)
                        
                        endPrint.endPrint(Re,XrS[0],Nu_AVG,t0)
                        
                        # # import matplotlib.pyplot as plt
                        # # plt.plot(xAll[nx1:],Nu_x[nx1:])
