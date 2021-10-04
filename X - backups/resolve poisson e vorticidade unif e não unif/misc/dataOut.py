#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 20:43:41 2021

@author: alegretti
"""
import os

def createFolderHydro(tag,ER,Re,Le,Ls,volNumb,tol,Xr,S,dtFac,phi,Pe,alpha,Pr,Ec,Nu,beta):
    
    pathHydro = "./0 - Results/{}_beta={}_ER={}_Re={:.2E}_Pr={:.2E}_Ec={:.2E}_phi={:.2E}_Pe={:.2E}_alpha={}_Le={}_Ls={}_Volumes={}_dtFac={}_tol={}_Xr={:.5f}_Nu={:.5f}".format(tag,beta,ER,Re,Pr,Ec,phi,Pe,alpha,Le,Ls,volNumb,dtFac,tol,Xr/S,Nu)
    os.makedirs(pathHydro)
    print('')
    print('Output folder created!')    
    os.chdir(pathHydro) 


def exportData(CVdata,field,mag,thermo,volNumb,nodeNumb,volNodes,nodesCoord):#,x_cv,thetaM,nu_x,nu_avg):
        
    myFile = open("FVfieldData.dat", 'w')
    myFile.write('VARIABLES="{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}" \n'.format("x","y","u","v","psi","w","Mx","My","alpha","Theta","MxH"))
    myFile.write('ZONE NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, VARLOCATION=([1,2]=nodal,[3,4,5,6,7,8,9,10,11]=CELLCENTERED), ZONETYPE=FEQUADRILATERAL\n'.format(nodeNumb,volNumb))

    for i in range(nodeNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(nodesCoord[i,1]))
        else:
            myFile.write('{} '.format(nodesCoord[i,1]))
    myFile.write('\n')    

    for i in range(nodeNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(nodesCoord[i,2]))
        else:
            myFile.write('{} '.format(nodesCoord[i,2]))
    myFile.write('\n')    

    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(field[i,1]))
        else:
            myFile.write('{} '.format(field[i,1]))
    myFile.write('\n')     

    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(field[i,2]))
        else:
            myFile.write('{} '.format(field[i,2]))
    myFile.write('\n')   

    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(field[i,3]))
        else:
            myFile.write('{} '.format(field[i,3]))
    myFile.write('\n')    

    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(field[i,4]))
        else:
            myFile.write('{} '.format(field[i,4]))
    myFile.write('\n')       
    
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(mag[i,5]))
        else:
            myFile.write('{} '.format(mag[i,5]))
    myFile.write('\n')  
    
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(mag[i,6]))
        else:
            myFile.write('{} '.format(mag[i,6]))
    myFile.write('\n')       
    
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(thermo[i,-1]))
        else:
            myFile.write('{} '.format(thermo[i,-1]))
    myFile.write('\n')       
    
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(thermo[i,1]))
        else:
            myFile.write('{} '.format(thermo[i,1]))
    myFile.write('\n')  
    
    ###
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(mag[i,4]*mag[i,5]))
        else:
            myFile.write('{} '.format(mag[i,4]*mag[i,5]))
    myFile.write('\n')  
    
    for i in range(volNumb):
        myFile.write('{} {} {} {}\n'.format(1+volNodes[i,1],1+volNodes[i,2],1+volNodes[i,3],1+volNodes[i,4]))
    myFile.close()    
    
    ##################  Exporting thetaM file
    
    # myFileTheta =  open("heatTransf.dat", 'w')
    # myFileTheta.write('Variables = "thetaM","Nu_x","x" \n')
    # myFileTheta.write('\n')
    # for i in range(len(x_cv)):
    #     myFileTheta.write('{:.3f}\t {:.3f}\t {:.3f}\n'.format(thetaM[i],nu_x[i],x_cv[i]))
    
    # myFileTheta.write('\n')
    # myFileTheta.write('Nu_AVG = {:.3f}'.format(nu_avg))    
    # myFileTheta.close()
    
    os.chdir('../../../') 

# def exportData(CVdata,field,mag,thermo,volNumb,nodeNumb,volNodes,nodesCoord,nx1,nx2,nx3,ny1,ny2,ny3):#,x_cv,thetaM,nu_x,nu_avg):
        
#     myFile = open("FVfieldData.dat", 'w')
#     myFile.write('VARIABLES="{}","{}","{}","{}","{}","{}","{}","{}","{}" \n'.format("x","y","u","v","psi","w","Mx","My","Theta"))
#     myFile.write('ZONE NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, VARLOCATION=([1,2]=nodal,[3,4,5,6,7,8,9]=CELLCENTERED), ZONETYPE=FEQUADRILATERAL\n'.format(nodeNumb,volNumb))

#     for i in range(nodeNumb):
#         if i % 1000 == 0:
#             myFile.write('{} \n'.format(nodesCoord[i,1]))
#         else:
#             myFile.write('{} '.format(nodesCoord[i,1]))
#     myFile.write('\n')    

#     for i in range(nodeNumb):
#         if i % 1000 == 0:
#             myFile.write('{} \n'.format(nodesCoord[i,2]))
#         else:
#             myFile.write('{} '.format(nodesCoord[i,2]))
#     myFile.write('\n')    

#     for i in range(volNumb):
#         if i % 1000 == 0:
#             myFile.write('{} \n'.format(field[i,1]))
#         else:
#             myFile.write('{} '.format(field[i,1]))
#     myFile.write('\n')     

#     for i in range(volNumb):
#         if i % 1000 == 0:
#             myFile.write('{} \n'.format(field[i,2]))
#         else:
#             myFile.write('{} '.format(field[i,2]))
#     myFile.write('\n')   

#     for i in range(volNumb):
#         if i % 1000 == 0:
#             myFile.write('{} \n'.format(field[i,3]))
#         else:
#             myFile.write('{} '.format(field[i,3]))
#     myFile.write('\n')    

#     for i in range(volNumb):
#         if i % 1000 == 0:
#             myFile.write('{} \n'.format(field[i,4]))
#         else:
#             myFile.write('{} '.format(field[i,4]))
#     myFile.write('\n')       
    
#     for i in range(volNumb):
#         if i % 1000 == 0:
#             myFile.write('{} \n'.format(mag[i,5]))
#         else:
#             myFile.write('{} '.format(mag[i,5]))
#     myFile.write('\n')  
    
#     for i in range(volNumb):
#         if i % 1000 == 0:
#             myFile.write('{} \n'.format(mag[i,6]))
#         else:
#             myFile.write('{} '.format(mag[i,6]))
#     myFile.write('\n')       
    
#     for i in range(volNumb):
#         if i % 1000 == 0:
#             myFile.write('{} \n'.format(thermo[i,1]))
#         else:
#             myFile.write('{} '.format(thermo[i,1]))
#     myFile.write('\n')  
    
#     for i in range(volNumb):
#         myFile.write('{} {} {} {}\n'.format(1+volNodes[i,1],1+volNodes[i,2],1+volNodes[i,3],1+volNodes[i,4]))
#     myFile.close()    
    
#     ##################  Exporting thetaM file
    
#     # myFileTheta =  open("heatTransf.dat", 'w')
#     # myFileTheta.write('Variables = "thetaM","Nu_x","x" \n')
#     # myFileTheta.write('\n')
#     # for i in range(len(x_cv)):
#     #     myFileTheta.write('{:.3f}\t {:.3f}\t {:.3f}\n'.format(thetaM[i],nu_x[i],x_cv[i]))
    
#     # myFileTheta.write('\n')
#     # myFileTheta.write('Nu_AVG = {:.3f}'.format(nu_avg))    
#     # myFileTheta.close()
    
    
    
#     os.chdir('../../../') 