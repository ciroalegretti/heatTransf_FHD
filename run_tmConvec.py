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
from numerics.vorticity import bc_vort,vort_cgs
from numerics.energy import bc_theta,energy_cgs
from numerics.poisson import bc_uvPsi,uAndV,poisson_cg
from post import calcNu,dataOut,profilesOut
import numpy as np
import os
# =============================================================================
volNumbArr = [10201]#[1681,6561,10201,25921]
tol = 1E-8
L = 1.0                 # Cavity width
h = 1.0                 # Cavity height
dtFac = 50.             # Time step coefficient
# tol = 1E-5            # Numerical tolerance for the outer iteration
Pr = 30.0               # Prandtl number

beta_dtArr = [0.3]#np.linspace(0.1,0.5,5)#[0.3 for U != 0]           # beta_dt = delta T/T_C
# =============================================================================
#                  DEFINE VARIABLE COEFFICIENTS AS ARRAYS 
# =============================================================================
phiArr = [1E-5]#np.linspace(0.06,0.01,6)             # Particle concentration  
# phiArr = np.insert(phiArr,0,1E-5)
alphaArr = [1E-5]#np.logspace(-1,1,3)                # Normalized magnetic field 
# alphaArr = np.insert(alphaArr,0,1E-5)
# alphaArr = alphaArr[15:]
PeArr = [1E+5]#[1E+5,1.0, 0.1, 0.01]               # Péclet number
Ec = 1E-6#np.logspace(-3,-6,4)           # Eckert number 
ReArr = [100.,]#400.,1000.]             # Reynolds number
lambdaArr = [0.0]#np.linspace(0,10,5)         # Lambda parameter for chain formation (Ivanov - M0)  

outTag = 'newValid' 
# =====================================================================
#            Define external field H, lid velocity and temperature BC
# =====================================================================
# applied field orientation
# magnet = 'vert'
magnet = 'horiz'

fieldSty = 'clegg'
# fieldSty = 'gradUnif'
# fieldSty = 'unif'

U = 1                   # Positive or negative lid velocity
# =============================================================================
# 
#                               MAIN LOOP 
# 
# =============================================================================
for phi in phiArr:
    for Pe in PeArr:
        for alpha0 in alphaArr:
            for Re in ReArr:
                for beta_dt in beta_dtArr:
                    for lamb in lambdaArr:
                        for volNumb in volNumbArr:
                            # for tol in tolArr:
                            
                            # Grid Import                
                            nx1,ny1,volArr,volNodes,nodesCoord,x1,y1,CVdata,volNumb,nodeNumb,L,h,\
                                                  wallFunc = loadGrid.loadGrid_LDC(volNumb,L,h)
    
                            # =====================================================================
                            eh = externalField.inputField(magnet)    
                            folderTag = '{}_{}_{}'.format(outTag,magnet,fieldSty)
                            x0,y0,b,Lm = externalField.getCoordLDC(eh,L)
                            dt = defTimeStep.dtMin(Re,Pr,dtFac,CVdata)
                            h_data,m_data,t_data,modH = startArrays.starters1d_LDC(fieldSty,volArr\
                                                    ,CVdata,alpha0,h,L,eh,x0,y0,b,Lm,phi,beta_dt,U)
    
                            i,l2_psi,l2_w,l2_theta,energy_imbalance =\
                                                0,1.0,1.0,1.0,1.0
                            i,resPsi,resW,resTheta = 0,1.0,1.0,1.0
    
                            # myfile = open('./Re={}_R2_{}vols_tol{:.1E}.dat'.format(Re,volNumb,tol),'w')
                            # myfile.write('Variables="it","resW","resPsi","resTheta" \n')
                            # =====================================================================
                            while max(resW,resPsi,resTheta,energy_imbalance) > tol or energy_imbalance > 1E-5:
                            # while max(resW,resPsi,resTheta) > tol:
                                
                                """Copying values before iteration"""
                                dataDummy = h_data.copy()   
                                m_dataDummy = m_data.copy()  
                                t_dataDummy = t_data.copy()
                                """Enforcing BC for stream function and velocity field"""
                                h_data = bc_uvPsi.bc_uvPsi_LDC(h_data,volArr,CVdata,h,U)
                                """Solving Poisson equation for streamfunction"""
                                h_data = poisson_cg.poisson_cgs(volNumb,volArr,CVdata,h_data,dt)
                                """Calculating velocity field components"""
                                h_data = uAndV.calculateUandV(h_data,volArr,CVdata)
                                """Enforcing BC for vorticity""" 
                                h_data = bc_vort.bc_w_LDC(h_data,volArr,CVdata,h,U)
                                """Solving vorticity equation"""
                                h_data = vort_cgs.vort_cgs(volNumb,wallFunc,volArr,CVdata,h_data,\
                                                            m_data,dt,Re,Pe,phi)
                                """Enforcing temperature BC"""
                                t_data = bc_theta.bc_theta_tmConv(t_data,volArr,CVdata)
                                """Solving energy equation with magnetic terms"""
                                t_data = energy_cgs.energy_cgs(volNumb,volArr,CVdata,wallFunc,\
                                                        h_data,t_data,m_data,phi,Ec,Re,Pe,Pr,dt)
                                """Updating alpha and eq. magnetization as temperature functions"""
                                # m_data,t_data = equilibriumMagIvanov.equilibriumMagIvanov(m_data,t_data,phi,\
                                #                                             alpha0,beta_dt,volNumb,lamb)
                                m_data,t_data = equilibriumMag.equilibriumMag(m_data,t_data,phi,alpha0,beta_dt,volNumb)
                                """Solving magnetization equation"""
                                m_data = solveMag.ftcsMx(h_data,m_data,volArr,CVdata,volNodes,\
                                                          nodesCoord,dt,phi,Pe)
                                m_data = solveMag.ftcsMy(h_data,m_data,volArr,CVdata,volNodes,\
                                                          nodesCoord,dt,phi,Pe)
                            
                                """Average Nu at left wall"""
                                Nu_in = calcNu.calcNu2_LDC(nx1,volArr,CVdata,volNumb,h_data,t_data,L,h,1,5,2)
                                """Average Nu at right wall"""
                                Nu_out = calcNu.calcNu2_LDC(nx1,volArr,CVdata,volNumb,h_data,t_data,L,h,3,5,2)
                                """Energy balance residual"""
                                energy_imbalance = np.abs(Nu_in - Nu_out)
                                # print(resNu)
                                
                                i+= 1

                                # l2_psi = np.sqrt(np.sum((h_data[:,3] - dataDummy[:,3])**2))
                                # l2_w = np.sqrt(np.sum((h_data[:,4] - dataDummy[:,4])**2))
                                # l2_theta = np.sqrt(np.sum((t_data[:,1] - t_dataDummy[:,1])**2))

                                resPsi = np.max(np.abs(h_data[:,3] - dataDummy[:,3]))
                                resW = np.max(np.abs(h_data[:,4] - dataDummy[:,4]))
                                resTheta = np.max(np.abs(t_data[:,1] - t_dataDummy[:,1]))
                                
                                if i % 1000 == 0:
                                
                                    print('{}k_it\t {:.2E}\t {:.2E}\t {:.2E}\t| \t {:.2E}'\
                                        .format(int(i/1000),resPsi,resW,resTheta,energy_imbalance))
                                    # print('{}k_it\t {:.2E}\t {:.2E}\t {:.2E} \t| \t {:.2E}'\
                                    #     .format(int(i/1000),l2_psi,l2_w,l2_theta,resNu))
                                
                                # myfile.write('{} \t {:.5E}\t {:.5E}\t {:.5E} \n'.format(i,erroW,erroPsi,erroTheta))
                                
                            # myfile.close()  
                            
                            
                            """ Post-processing and output"""
    
                            dataOut.createFolderHydro_LDC(folderTag,Re,volNumb,tol,dtFac,phi,\
                                                                            Pe,alpha0,Pr,Nu_in,i,h_data,lamb,Ec)

                            gradChi,t_data = dataOut.gradChi_LDC(t_data, m_data, volNumb, volArr, CVdata)

                            dataOut.exportData(CVdata,h_data,m_data,t_data,volNumb,nodeNumb,\
                                               volNodes,nodesCoord,modH,Nu_in)
                                
                            profilesOut.LDC_profiles(CVdata, h_data, m_data, t_data, volArr, volNumb)
                            
                            os.chdir('../../../') 
                            print('\n psiMin = {:.4f}'.format(np.min(h_data[:volNumb,3])))
                            print('\n Nu_avg = {:.4f}'.format(Nu_in))
                            print('')
                            print('Data exported!')    
                            print('')
                            print('')

        