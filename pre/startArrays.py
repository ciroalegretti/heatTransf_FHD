#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:26:16 2021

@author: alegretti
"""
import numpy as np
# from numba import jit
from pre import externalField
from numerics.magnetization import equilibriumMag
from numerics.energy import bc_theta
from numerics.poisson import bc_uvPsi

# def startArrays(model,fieldType,volArr,CVdata,alpha0,eh,L,Le,S,h):
# # =============================================================================
# #     
# #     Return 2D array data:
# #         
# #        data    = [#vol, u, v, psi, w]
# #        magData = [#vol, M0x, M0y, Hx, Hy, Mx, My]
# #        thermoData = [#vol, theta, chi, alphaLang]
# #        
# # =============================================================================

#     n = len(volArr) # internal volumes
#     lastGhost = np.min(volArr) 
#     gi = np.abs(lastGhost) # last ghost abs index
#     nt = n + gi    # total internal and ghost volumes
    
#     data = np.zeros((nt,5))
#     magData = np.zeros((nt,7))
#     thermoData = np.zeros((nt,4))
    
#     for i in range(n):
#         data[i,0] = CVdata[i,0]
#         magData[i,0] = CVdata[i,0]
#         thermoData[i,0] = CVdata[i,0]
#         thermoData[i,-1] = alpha0
        
#         data[i,3] = 0.1
        
#     ### Initialize magnetic         
#     magData[:,3],magData[:,4],H0 = externalField.externalField(model,fieldType\
#                                                 ,eh,magData,n,CVdata,L,Le,S)
    
#     return data,magData,thermoData,H0

# def starters1d(volArr,CVdata,Hx,Hy,alpha0,S,Le,Ls):
# # =============================================================================
# #     
# #     Return 2D array data:
# #         
# #        data    = [#vol, u, v, psi, w]
# #        magData = [#vol, M0x, M0y, Hx, Hy, Mx, My]
# #        thermoData = [#vol, theta, chi, alphaLang]
# #        
# # =============================================================================
    
#     n = len(volArr) # internal volumes
#     lastGhost = np.min(volArr) 
#     gi = np.abs(lastGhost) # last ghost abs index
#     nt = n + gi    # total internal and ghost volumes
    
#     data = np.zeros((nt,5))
#     magData = np.zeros((nt,7))
#     thermoData = np.zeros((nt,4))
    
#     # fluxes arrays
#     flux = np.zeros((n,4))
    
#     for i in range(n):
#         data[i,0] = CVdata[i,0]
#         magData[i,0] = CVdata[i,0]
#         thermoData[i,0] = CVdata[i,0]
#         thermoData[i,-1] = alpha0
        
#     ### Clegg permanent magnetic applied field
#     magData[:,3],magData[:,4] = clegg.cleggStepFace(magData,n,CVdata,S,Le)
#     # magData[:,3],magData[:,4] = clegg.cleggStepBW(magData,n,CVdata,S,Le,Ls)
    
#     ### Uniform applied field
#     # magData[:,3] = Hx
#     # magData[:,4] = Hy
    
#     return data,magData,thermoData,flux

# def starters1d_PP(volArr,CVdata,Hx,Hy,alpha0,h,Le):
#     """ Return 2D array data:
        
#        data    = [#vol, u, v, psi, w]
#        magData = [#vol, M0x, M0y, Hx, Hy, Mx, My]
#        thermoData = [#vol, theta, chi, alphaLang]
       
#        flux = [E,N,W,S]   -> fluxes through each face
        
#     """
#     n = len(volArr) # internal volumes
#     lastGhost = np.min(volArr) 
#     gi = np.abs(lastGhost) # last ghost abs index
#     nt = n + gi    # total internal and ghost volumes
    
#     data = np.zeros((nt,5))
#     magData = np.zeros((nt,7))
#     thermoData = np.zeros((nt,4))

    
#     # fluxes arrays
#     flux = np.zeros((n,4))
    
#     for i in range(n):
#         data[i,0] = CVdata[i,0]
#         magData[i,0] = CVdata[i,0]
#         thermoData[i,0] = CVdata[i,0]
#         thermoData[i,-1] = alpha0
        
#     ### Uniform applied field
#     # magData[:,3] = Hx
#     # magData[:,4] = Hy
    
#     ### Clegg permanent magnetic applied field
#     magData[:,3],magData[:,4] = clegg.cleggPP(magData,n,CVdata,h,Le)
    
#     return data,magData,thermoData,flux

def starters1d_LDC(fieldType,volArr,CVdata,alpha0,h,Le,eh,x0,y0,b,Lm,phi,beta,U):
# =============================================================================
#     
#     Return 2D array data:
#         
#        data    = [#vol, u, v, psi, w]
#        magData = [#vol, M0x, M0y, Hx, Hy, Mx, My]
#        thermoData = [#vol, theta, chi, alphaLang]
#        
# =============================================================================

    volNumb = len(volArr) # internal volumes
    lastGhost = np.min(volArr) 
    gi = np.abs(lastGhost) # last ghost abs index
    nt = volNumb + gi    # total internal and ghost volumes
    
    data = np.zeros((nt,5))
    magData = np.zeros((nt,7))
    thermoData = np.zeros((nt,4))
    
    for i in range(volNumb):
        data[i,0] = CVdata[i,0]
        magData[i,0] = CVdata[i,0]
        thermoData[i,0] = CVdata[i,0]
        thermoData[i,-1] = alpha0
        
        data[i,3] = 0.1
        
    #     # fluxes arrays
    # flux = np.zeros((n,4))
        
    ### Set external magnetic field
    magData[:,3],magData[:,4],H0 = externalField.externalField(fieldType,eh,magData,volNumb,CVdata,x0,y0,b,Lm)
    
    #  Setting initial magnetization as M0
    magData,thermoData = equilibriumMag.equilibriumMag(magData,thermoData,phi,alpha0,beta,volNumb)

    # Applying hydrodynamic and thermal BC
    data = bc_uvPsi.bc_uvPsi_LDC(data,volArr,CVdata,h,U)
    thermoData = bc_theta.bc_theta_LDC(thermoData,volArr,CVdata)
    
    return data,magData,thermoData,H0

def starters1d_PP(fieldType,volArr,CVdata,alpha0,h,Le,eh,x0,y0,b,Lm,phi,beta):
# =============================================================================
#     
#     Return 2D array data:
#         
#        data    = [#vol, u, v, psi, w]
#        magData = [#vol, M0x, M0y, Hx, Hy, Mx, My]
#        thermoData = [#vol, theta, chi, alphaLang]
#        
# =============================================================================

    volNumb = len(volArr) # internal volumes
    lastGhost = np.min(volArr) 
    gi = np.abs(lastGhost) # last ghost abs index
    nt = volNumb + gi    # total internal and ghost volumes
    
    data = np.zeros((nt,5))
    magData = np.zeros((nt,7))
    thermoData = np.zeros((nt,4))
    
    for i in range(volNumb):
        data[i,0] = CVdata[i,0]
        magData[i,0] = CVdata[i,0]
        thermoData[i,0] = CVdata[i,0]
        thermoData[i,-1] = alpha0
        
        data[i,3] = 0.1
        
    ### Set external magnetic field
    magData[:,3],magData[:,4],H0 = externalField.externalField(fieldType,eh,magData,volNumb,CVdata,x0,y0,b,Lm)
    
    #  Setting initial magnetization as M0
    magData,thermoData = equilibriumMag.equilibriumMag(magData,thermoData,phi,alpha0,beta,volNumb)

    # Applying hydrodynamic and thermal BC
    data = bc_uvPsi.bc_uvPsi_PP(data,volArr,CVdata,h)
    thermoData = bc_theta.bc_theta_PP(thermoData,volArr,CVdata)
    
    return data,magData,thermoData,H0

def starters1d_BFS(fieldType,volArr,CVdata,alpha0,h,Le,eh,x0,y0,b,Lm,phi,beta):
# =============================================================================
#     
#     Return 2D array data:
#         
#        data    = [#vol, u, v, psi, w]
#        magData = [#vol, M0x, M0y, Hx, Hy, Mx, My]
#        thermoData = [#vol, theta, chi, alphaLang]
#        
# =============================================================================

    volNumb = len(volArr) # internal volumes
    lastGhost = np.min(volArr) 
    gi = np.abs(lastGhost) # last ghost abs index
    nt = volNumb + gi    # total internal and ghost volumes
    
    data = np.zeros((nt,5))
    magData = np.zeros((nt,7))
    thermoData = np.zeros((nt,4))
    
    for i in range(volNumb):
        data[i,0] = CVdata[i,0]
        magData[i,0] = CVdata[i,0]
        thermoData[i,0] = CVdata[i,0]
        thermoData[i,-1] = alpha0
        
        data[i,3] = 0.1
        
    ### Set external magnetic field
    magData[:,3],magData[:,4],H0 = externalField.externalField(fieldType,eh,magData,volNumb,CVdata,x0,y0,b,Lm)
    
    #  Setting initial magnetization as M0
    magData,thermoData = equilibriumMag.equilibriumMag(magData,thermoData,phi,alpha0,beta,volNumb)

    # Applying hydrodynamic and thermal BC
    data = bc_uvPsi.bc_uvPsi_PP(data,volArr,CVdata,h)
    thermoData = bc_theta.bc_theta_PP(thermoData,volArr,CVdata)
    
    return data,magData,thermoData,H0