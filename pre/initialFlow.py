#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:25:08 2022

@author: alegretti

Solves steady state flow in the absence of magnetic effects as inital condition to further develop the coupled problem with thermal and magnetic effects to achieve coupled steady state solution

"""
import numpy as np
from numerics.poisson import bc_uvPsi,poisson_cg,uAndV
from numerics.vorticity import bc_vort,vort_cgs
from post import calcXr,dataOut
import os

def initialSeek(model,Re,volNumb,volNodes,nodesCoord,nodeNumb,CVdata,volArr,wallFunc,ER,S,edgeVol,L,Le,h,dt,nx3,y1,beta,h_data,folderTag):
    
    if model == '1':
        hydroResCheck = os.path.exists("./initial/LDC/Re={:.1E}_Volumes={}".format(Re,volNumb))
        
        if hydroResCheck == False:
            h_data = initialFlow(model,CVdata,volNumb,volNodes,nodeNumb,nodesCoord,beta,volArr,wallFunc,h_data,S,ER,folderTag,edgeVol,Re,dt,Le,L,h,y1,nx3)
            print("\n Hydrodynamic solution computed and stored, starting coupled solution \n")
        else:
            h_data = np.load("./initial/LDC/Re={:.1E}_Volumes={}".format(Re,volNumb))['h_data']
            print('\n Available hydrodynamic solution imported, starting coupled solution \n')
    
    elif model == '2':
        hydroResCheck = os.path.exists("./initial/PP/Re={:.1E}_Volumes={}".format(Re,volNumb))
        
        if hydroResCheck == False:
            h_data = initialFlow(model,CVdata,volNumb,volNodes,nodeNumb,nodesCoord,beta,volArr,wallFunc,h_data,S,ER,folderTag,edgeVol,Re,dt,Le,L,h,y1,nx3)
            print("\n Hydrodynamic solution computed and stored, starting coupled solution \n")
        else:
            h_data = np.load("./initial/PP/Re={:.1E}_Volumes={}".format(Re,volNumb))['h_data']
            print('\n Available hydrodynamic solution imported, starting coupled solution \n')
    
    elif model == '3':
        hydroResCheck = os.path.exists("./initial/BFS/beta={}_ER={}_Re={:.1E}_Le={}_Ls={}_Volumes={}".format(beta,ER,Re,Le,L,volNumb))
        
        if hydroResCheck == False:
            h_data = initialFlow(model,CVdata,volNumb,volNodes,nodeNumb,nodesCoord,beta,volArr,wallFunc,h_data,S,ER,folderTag,edgeVol,Re,dt,Le,L,h,y1,nx3)
            print("\n Hydrodynamic solution computed and stored, starting coupled solution \n")
        else:
            h_data = np.load("./initial/BFS/beta={}_ER={}_Re={:.1E}_Le={}_Ls={}_Volumes={}/hydroSol.npz".format(beta,ER,Re,Le,L,volNumb))['h_data']
            print('\n Available hydrodynamic solution imported, starting coupled solution \n')

    return h_data

def initialFlow(model,CVdata,volNumb,volNodes,nodeNumb,nodesCoord,beta,volArr,wallFunc,h_data,S,ER,label,edgeVol,Re,dt,Le,L,h,y1,nx3):

    print("\n No solution available, starting hydrodynamic computations \n")

    tol = 1E-4
    i = 0
    erroPsi = 1.0
    erroW = 1.0
    
    if model == '1':
        # myfile = open('./initial/LDC/convXr_{}vols_tol{:.1E}.dat'.format(volNumb,tol),'w')
        # myfile.write('Variables="it","Xr"\n')
    
        while max(erroW,erroPsi) > tol:
    
            """Copying values before iteration"""
            dataDummy = h_data.copy()   
            """Enforcing BC for stream function and velocity field"""
            h_data = bc_uvPsi.bc_uvPsi_LDC(h_data,volArr,CVdata,h)
            """Solving Poisson equation for streamfunction"""
            # h_data = poisson_cg.poisson2_cgs(edgeVol,volNumb,volArr,CVdata,h_data,ER,h,dt)
            h_data = poisson_cg.poisson_cgs(volNumb,volArr,CVdata,h_data,dt)
            # h_data = eulerExp.poissonExplicit(h_data,volArr,volNodes,CVdata)
            """Calculating velocity field components"""
            h_data = uAndV.calculateUandV(h_data,volArr,CVdata)
            """Enforcing BC for vorticity""" 
            h_data = bc_vort.bc_w_LDC(h_data,volArr,CVdata,h)
            """Solving vorticity equation"""
            h_data = vort_cgs.vort_cgs(volNumb,wallFunc,volArr,CVdata,h_data,h_data,dt,Re,1.0,phi=0.0)
            # h_data = macCormack_w.macCormack_w(h_data,m_data,Re,flux,wallFunc,volArr,CVdata,dt,Pe,phi)
    
            erroPsi = np.max(np.abs(h_data[:,3] - dataDummy[:,3]))
            erroW = np.max(np.abs(h_data[:,4] - dataDummy[:,4]))
    
            # myfile.write('{} \t {:.5E} \n'.format(i,XrS))
    
            i+= 1
            # print(XrS)
            if i % 1000 == 0:
                print('{}k_it\t {:.2E}\t {:.2E}'.format(int(i/1000),erroPsi,erroW))
    
        # myfile.close()
    
        dataOut.exportHydro(model,beta,ER,Re,Le,L,volNumb,h_data,nodeNumb,nodesCoord,volNodes)

    if model == '2':
        # myfile = open('./initial/LDC/convXr_{}vols_tol{:.1E}.dat'.format(volNumb,tol),'w')
        # myfile.write('Variables="it","Xr"\n')
    
        while max(erroW,erroPsi) > tol:
    
            """Copying values before iteration"""
            dataDummy = h_data.copy()   
            """Enforcing BC for stream function and velocity field"""
            h_data = bc_uvPsi.bc_uvPsi_PP(h_data,volArr,CVdata,h)
            """Solving Poisson equation for streamfunction"""
            # h_data = poisson_cg.poisson2_cgs(edgeVol,volNumb,volArr,CVdata,h_data,ER,h,dt)
            h_data = poisson_cg.poisson_cgs(volNumb,volArr,CVdata,h_data,dt)
            # h_data = eulerExp.poissonExplicit(h_data,volArr,volNodes,CVdata)
            """Calculating velocity field components"""
            h_data = uAndV.calculateUandV(h_data,volArr,CVdata)
            """Enforcing BC for vorticity""" 
            h_data = bc_vort.bc_w_PP(h_data,volArr,CVdata,h)
            """Solving vorticity equation"""
            h_data = vort_cgs.vort_cgs(volNumb,wallFunc,volArr,CVdata,h_data,h_data,dt,Re,1.0,phi=0.0)
            # h_data = macCormack_w.macCormack_w(h_data,m_data,Re,flux,wallFunc,volArr,CVdata,dt,Pe,phi)
    
            erroPsi = np.max(np.abs(h_data[:,3] - dataDummy[:,3]))
            erroW = np.max(np.abs(h_data[:,4] - dataDummy[:,4]))
    
            # myfile.write('{} \t {:.5E} \n'.format(i,XrS))
    
            i+= 1
            # print(XrS)
            if i % 1000 == 0:
                print('{}k_it\t {:.2E}\t {:.2E}'.format(int(i/1000),erroPsi,erroW))
    
        # myfile.close()
    
        dataOut.exportHydro(model,beta,ER,Re,Le,L,volNumb,h_data,nodeNumb,nodesCoord,volNodes)


    elif model == '3':   # BFS

        myfile = open('./initial/BFS/convXr_{}vols_tol{:.1E}.dat'.format(volNumb,tol),'w')
        myfile.write('Variables="it","Xr"\n')
    
        while max(erroW,erroPsi) > tol:
    
            """Copying values before iteration"""
            dataDummy = h_data.copy()   
            """Enforcing BC for stream function and velocity field"""
            h_data = bc_uvPsi.bc_uvPsi(h_data,volArr,CVdata,edgeVol,S)
            """Solving Poisson equation for streamfunction"""
            # h_data = poisson_cg.poisson2_cgs(edgeVol,volNumb,volArr,CVdata,h_data,ER,h,dt)
            h_data = poisson_cg.poisson_cgs(volNumb,volArr,CVdata,h_data,dt)
            # h_data = eulerExp.poissonExplicit(h_data,volArr,volNodes,CVdata)
            """Calculating velocity field components"""
            h_data = uAndV.calculateUandV(h_data,volArr,CVdata)
            """Enforcing BC for vorticity""" 
            h_data = bc_vort.bc_w(h_data,volArr,CVdata,edgeVol,S)
            """Solving vorticity equation"""
            h_data = vort_cgs.vort_cgs(volNumb,wallFunc,volArr,CVdata,h_data,h_data,dt,Re,1.0,phi=0.0)
            # h_data = macCormack_w.macCormack_w(h_data,m_data,Re,flux,wallFunc,volArr,CVdata,dt,Pe,phi)
    
            erroPsi = np.max(np.abs(h_data[:,3] - dataDummy[:,3]))
            erroW = np.max(np.abs(h_data[:,4] - dataDummy[:,4]))
    
            XrS = calcXr.calcXr(edgeVol,volArr,h_data,CVdata,Le,y1,S,nx3)
    
            myfile.write('{} \t {:.5E} \n'.format(i,XrS))
    
            i+= 1
            # print(XrS)
            if i % 1000 == 0:
                print('{}k_it\t {:.2E}\t {:.2E}\t {:.3f}'.format(int(i/1000),erroPsi,erroW,XrS))
    
        myfile.close()
    
        dataOut.exportHydro(model,beta,ER,Re,Le,L,volNumb,h_data,nodeNumb,nodesCoord,volNodes)

    return h_data
