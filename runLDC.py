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
from numerics.magnetization import equilibriumMag,solveMag,equilibriumMagIvanov,mag_cgs
from numerics.vorticity import bc_vort,vort_cgs
from numerics.energy import bc_theta,energy_cgs
from numerics.poisson import bc_uvPsi,uAndV,poisson_cg
from post import calcNu,dataOut,profilesOut
import numpy as np
import os
# =============================================================================
volNumbArr = [6561]#[1681,6561,10201,25921]
L = 1.0                 # Cavity width
h = 1.0                 # Cavity height
dtFac = 10.             # Time step coefficient
tol = 1E-8              # Numerical tolerance for the outer iteration
Pr = 30.0               # Prandtl number
beta_dt = 0.3           # beta_dt = delta T/T_C
EcArr = [1E-6]#np.logspace(-3,-6,4)                # Eckert number 
alphaArr = [1.0]#np.logspace(-1,1,3)                # Normalized magnetic field 
ReArr = [100.]#[100.,400.,1000.]                   # Reynolds number
# =============================================================================
#                  DEFINE VARIABLE COEFFICIENTS AS ARRAYS 
# =============================================================================
phiArr = [0.2]#np.linspace(0.06,0.01,6)             # Particle concentration  
PeArr = [0.01]#[1E+5,1.0, 0.1, 0.01]               # Péclet number
lambdaArr = [0.0]#np.linspace(0,10,5)              # Lambda parameter for chain formation (Ivanov - M0)  

outTag = 'transient_strongMag'
# =====================================================================
#            Define external field H, lid velocity and temperature BC
# =====================================================================
# applied field orientation
magnet = 'vert'
# magnet = 'horiz'

fieldSty = 'clegg'
# fieldSty = 'gradUnif'
# fieldSty = 'unif'

U = 1                   # Positive or negative lid velocity
# =============================================================================
# 
#                               MAIN5 LOOP 
# 
# =============================================================================
for phi in phiArr:
    for Pe in PeArr:
        for alpha0 in alphaArr:
            for Re in ReArr:
                for Ec in EcArr:
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
                            # while max(resW,resPsi,resTheta) > tol:# or energy_imbalance > 1E-4:
                            # while max(resW,resPsi,resTheta) > tol:
                            while i < 500000:
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
                                h_data = vort_cgs.vort_cgs_LDC(volNumb,wallFunc,volArr,CVdata,h_data,\
                                                            m_data,t_data,dt,Re,Pe,phi)
                                """Enforcing temperature BC"""
                                t_data = bc_theta.bc_theta_LDC(t_data,volArr,CVdata)
                                """Solving energy equation with magnetic terms"""
                                t_data = energy_cgs.energy_cgs(volNumb,volArr,CVdata,wallFunc,\
                                                        h_data,t_data,m_data,phi,Ec,Re,Pe,Pr,dt)
                                """Updating alpha and eq. magnetization as temperature functions"""
                                m_data,t_data = equilibriumMagIvanov.equilibriumMagIvanov(m_data,t_data,phi,\
                                                                            alpha0,beta_dt,volNumb,lamb)
                                # m_data,t_data = equilibriumMag.equilibriumMag(m_data,t_data,phi,\
                                #                                             alpha0,beta_dt,volNumb)
                                """Solving magnetization equation"""
                                ### _______FTCS______ (high Pe req) ###
                                m_data = solveMag.ftcsMx(h_data,m_data,t_data,volArr,CVdata,volNodes,\
                                                          nodesCoord,dt,phi,Pe)
                                m_data = solveMag.ftcsMy(h_data,m_data,t_data,volArr,CVdata,volNodes,\
                                                          nodesCoord,dt,phi,Pe)
                                ###_________________________________###
                                ###________CGS______________________###
                                # m_data = mag_cgs.magY_cgs(volNumb,wallFunc,volArr,CVdata,h_data,\
                                #                             m_data,t_data,dt,Pe,phi)
                                # m_data = mag_cgs.magX_cgs(volNumb,wallFunc,volArr,CVdata,h_data,\
                                #                             m_data,t_data,dt,Pe,phi)

                               

                                i+= 1

                                # l2_psi = np.sqrt(np.sum((h_data[:,3] - dataDummy[:,3])**2))
                                # l2_w = np.sqrt(np.sum((h_data[:,4] - dataDummy[:,4])**2))
                                # l2_theta = np.sqrt(np.sum((t_data[:,1] - t_dataDummy[:,1])**2))

                                resTheta = np.max(np.abs(t_data[:,1] - t_dataDummy[:,1]))
                                resPsi = np.max(np.abs(h_data[:,3] - dataDummy[:,3]))
                                resW = np.max(np.abs(h_data[:,4] - dataDummy[:,4]))
                                resMx = np.max(np.abs(m_data[:,-2] - m_dataDummy[:,-2]))
                                resMy = np.max(np.abs(m_data[:,-1] - m_dataDummy[:,-1]))

                                # print('{}k_it\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}'\
                                #     .format(int(i/1000),resPsi,resW,resTheta,resMx,resMy))

                                if i % 1000 == 0 and i>0:

                                    """Average Nu at bottom wall"""
                                    Nu_in = calcNu.calcNu2_LDC(nx1,volArr,CVdata,volNumb,h_data,t_data,L,h,2,4,3)
                                    """Average Nu at top wall"""
                                    Nu_out = calcNu.calcNu2_LDC(nx1,volArr,CVdata,volNumb,h_data,t_data,L,h,4,4,3)
                                    # """Average Nu at left wall"""
                                    # Nu_out = calcNu.calcNu2_LDC(ny1,volArr,CVdata,volNumb,h_data,t_data,L,h,1,5,2)
                                    # """Average Nu at right wall"""
                                    # Nu_in = calcNu.calcNu2_LDC(ny1,volArr,CVdata,volNumb,h_data,t_data,L,h,3,5,2)

                                    """Energy balance residual"""
                                    energy_imbalance = np.abs(Nu_in - Nu_out)

                                    print('{}k_it\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}\t {:.2E}\t| \t {:.2E}'\
                                        .format(int(i/1000),resPsi,resW,resTheta,resMx,resMy,energy_imbalance))
                                    # print('{}k_it\t {:.2E}\t {:.2E}\t {:.2E} \t| \t {:.2E}'\
                                    #     .format(int(i/1000),l2_psi,l2_w,l2_theta,resNu))

                                # myfile.write('{} \t {:.5E}\t {:.5E}\t {:.5E} \n'.format(i,erroW,erroPsi,erroTheta))

                            # myfile.close()  


                                    """ Post-processing and output"""
                                    """Enforcing BC for stream function and velocity field"""
                                    h_data = bc_uvPsi.bc_uvPsi_LDC(h_data,volArr,CVdata,h,U)
                                    """Enforcing BC for vorticity""" 
                                    h_data = bc_vort.bc_w_LDC(h_data,volArr,CVdata,h,U)
                                    """Enforcing temperature BC"""
                                    t_data = bc_theta.bc_theta_LDC(t_data,volArr,CVdata)                            
        
                                    dataOut.createFolderHydro_LDC(folderTag,Re,volNumb,tol,dtFac,phi,\
                                                                                    Pe,alpha0,Pr,Nu_in,i,h_data,lamb,Ec)
        
                                    gradChi,t_data = dataOut.gradChi_LDC(t_data, m_data, volNumb, volArr, CVdata)
        
                                    tm_torque = dataOut.calculate_tmTorque(volNumb, volArr, t_data, CVdata, modH)
                                    lap_MxH = dataOut.calc_lap_MxH(volNumb, volArr, m_data, CVdata)
        
                                    dataOut.exportData(CVdata,h_data,m_data,t_data,volNumb,nodeNumb,\
                                                       volNodes,nodesCoord,modH,Nu_in,tm_torque,lap_MxH)
        
                                    profilesOut.LDC_profiles(CVdata, h_data, m_data, t_data, volArr, volNumb)
        
                                    energyOut = open('energy_residual_imbalance','w')
                                    energyOut.write('Energy imbalance due to a residual heat source between non-adiabitic walls (Nusselt number difference; as Moallemi) \n \n')
                                    energyOut.write('{:.1E}'.format(energy_imbalance))
                                    energyOut.close()
        
                                    os.chdir('../../../')
                                    print('\n psiMin = {:.4f}'.format(np.min(h_data[:volNumb,3])))
                                    print('\n Nu_avg = {:.4f}'.format(Nu_in))
                                    print('')
                                    print('Data exported!')
                                    print('')
                                    print('')
