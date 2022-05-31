#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runLDC.py

Steady-state solutions of the Lid-Driven Cavity problem under vorticity-stream
function formulation with magnetic effects and heat transfer

Created on Wed Jul 28 19:07:43 2021

@author: alegretti
"""
from pre import loadGrid,defTimeStep,startArrays,externalField
from numerics.magnetization import equilibriumMag,equilibriumMagIvanov,solveMag
from numerics.vorticity import bc_vort,vort_cgs,macCormack_w
from numerics.energy import bc_theta,energy_cgs,macCormack_theta
from numerics.poisson import bc_uvPsi,uAndV,poisson_cg,eulerExp
from post import calcNu,dataOut,profilesOut
import numpy as np

# =============================================================================
volNumb = 10000 #[400,1600,3600,6400,10000,25600]
tol = 1E-6
L = 1.0                 # Cavity width
h = 1.0                 # Cavity height
dtFac = 1.              # Time step coefficient
# tol = 1E-5            # Numerical tolerance for the outer iteration
Pr = 1.                 # Prandtl number
Ec = 1E-6               # Eckert number 
beta_dt = 0.3           # beta_dt = delta T/T_C
# =============================================================================
#                  DEFINE VARIABLE COEFFICIENTS AS ARRAYS 
# =============================================================================
phiArr = np.linspace(0.01,0.10,10)
# phiArr = np.insert(phiArr,0,1E-5)
PeArr = [1.0,]#np.logspace(-2,2,5)#[1E+5]#[1E-2,1E-1,1E+0]#np.logspace(-3,0,4)
alphaArr = [10.]#[1.,]#np.logspace(-1,1,3)
# alphaArr = np.insert(alphaArr,0,1E-5)
# alphaArr = alphaArr[15:]
ReArr = [100.]#,400.,1000.,2000.]
# tolArr = [1E-6]#[1E-9,1E-10]#np.logspace(-3,-8,6)

lamb = 0                # Lambda parameter for chain formation (Ivanov - M0)

outTag = 'encit5_varrPhi'
# =====================================================================
#            Define external field H and start arrays
# =====================================================================
# magnet = 'horiz'
# magnet = 'vert'
magnetArr = ['vert']#['horiz','vert']

# fieldSty = 'clegg'
# fieldSty = 'gradUnif'
# fieldSty = 'unif'
fieldStyArr = ['gradUnif']#['clegg','gradUnif','unif']
# =============================================================================
# 
#                               MAIN LOOP 
# 
# =============================================================================
for Pe in PeArr:
    for alpha0 in alphaArr:
        for phi in phiArr:
            for Re in ReArr:
                for magnet in magnetArr:
                    for fieldSty in fieldStyArr:
                        
                        # Grid Import                
                        nx1,ny1,volArr,volNodes,nodesCoord,x1,y1,CVdata,volNumb,nodeNumb,L,h,\
                                              wallFunc = loadGrid.loadGrid_LDC(volNumb,L,h)

                        # =====================================================================
                        eh = externalField.inputField(magnet)    
                        folderTag = '{}_{}_{}'.format(outTag,magnet,fieldSty)
                        x0,y0,b,Lm = externalField.getCoordLDC(eh,L)
                        dt = defTimeStep.dtMin(Re,Pr,dtFac,CVdata)
                        h_data,m_data,t_data,modH = startArrays.starters1d_LDC(fieldSty,volArr\
                                                ,CVdata,alpha0,h,L,eh,x0,y0,b,Lm,phi,beta_dt)

                        i,erroPsi,erroW,erroMx,erroMy,erroTheta = 0,1.0,1.0,1.0,1.0,1.0

                        # myfile = open('./Re={}_R2_{}vols_tol{:.1E}.dat'.format(Re,volNumb,tol),'w')
                        # myfile.write('Variables="it","resW","resPsi","resTheta" \n')
                                                
                        # =====================================================================
                        while max(erroW,erroPsi,erroMx,erroMy,erroTheta) > tol:
                        
                            """Copying values before iteration"""
                            dataDummy = h_data.copy()   
                            m_dataDummy = m_data.copy()  
                            t_dataDummy = t_data.copy()
                            """Enforcing BC for stream function and velocity field"""
                            h_data = bc_uvPsi.bc_uvPsi_LDC(h_data,volArr,CVdata,h)
                            """Solving Poisson equation for streamfunction"""
                            h_data = eulerExp.poissonExplicit(h_data,volArr,volNodes,CVdata)
                            # h_data = poisson_cg.poisson_cgs(volNumb,volArr,CVdata,h_data,dt)
                            """Calculating velocity field components"""
                            h_data = uAndV.calculateUandV(h_data,volArr,CVdata)
                            """Enforcing BC for vorticity""" 
                            h_data = bc_vort.bc_w_LDC(h_data,volArr,CVdata,h)
                            """Solving vorticity equation"""
                            h_data = vort_cgs.vort_cgs(volNumb,wallFunc,volArr,CVdata,h_data,\
                                                        m_data,dt,Re,Pe,phi)
                            """Enforcing temperature BC"""
                            t_data = bc_theta.bc_theta_LDC(t_data,volArr,CVdata)
                            """Solving energy equation with magnetic terms"""
                            t_data = energy_cgs.energy_cgs(volNumb,volArr,CVdata,wallFunc,\
                                                    h_data,t_data,m_data,phi,Ec,Re,Pe,Pr,dt)
                            """Updating alpha and eq. magnetization as temperature functions"""
                            # m_data,t_data = equilibriumMag.equilibriumMag(m_data,t_data,phi,\
                            #                                             alpha0,beta_dt,volNumb)
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
                            
                            if i % 10000 == 0:
                                print('{}k_it\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}'\
                                    .format(int(i/1000),erroPsi,erroW,erroMx,erroMy,erroTheta))
                        
                        print('psiMin = {:.4f}'.format(np.min(h_data[:volNumb,3])))
                            
                            # myfile.write('{} \t {:.5E}\t {:.5E}\t {:.5E} \n'.format(i,erroW,erroPsi,erroTheta))
                            
                        # myfile.close()  
                        
                        """ Check thermal validation and Nu_AVG definition !!!!!"""
                        Tm,xAll,Nu_x,Nu_AVG,uEnt,uThetaEnt,dtBulk,arr2d = calcNu.calcNu2_LDC(nx1,ny1,\
                                               volArr,CVdata,volNumb,h_data,t_data,L,h)

                        dataOut.createFolderHydro_LDC(folderTag,Re,volNumb,tol,dtFac,phi,\
                                                                        Pe,alpha0,Pr,Nu_AVG,i,h_data,lamb)
                        dataOut.exportData(CVdata,h_data,m_data,t_data,volNumb,nodeNumb,\
                                           volNodes,nodesCoord,modH,xAll,Tm,Nu_x,Nu_AVG)
                            
                        profilesOut.LDC_lidVort(nx1,Re,arr2d,CVdata,h_data,volArr)
