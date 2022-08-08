#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 20:43:41 2021

@author: alegretti
"""
import os
import numpy as np

def exportHydro(model,beta,ER,Re,Le,Ls,volNumb,h_data,nodeNumb,nodesCoord,volNodes):
    
    """ Export steady state pure hydrodynamic solution  """    
    
    if model == '1':
        pathHydro = "./initial/LDC/Re={:.1E}_Volumes={}".format(Re,volNumb)    
    elif model == '2':
        pathHydro = "./initial/PP/Re={:.1E}_Volumes={}".format(Re,volNumb)    
    elif model == '3':
        pathHydro = "./initial/BFS/beta={}_ER={}_Re={:.1E}_Le={}_Ls={}_Volumes={}".format(beta,ER,Re,Le,Ls,volNumb)
   
    os.makedirs(pathHydro)
    os.chdir(pathHydro) 
    np.savez('hydroSol.npz', h_data=h_data)
    
    myFile = open("FVfieldData.dat", 'w')
    myFile.write('VARIABLES="{}","{}","{}","{}","{}","{}" \n'.format("x","y","u","v","psi","w"))
    myFile.write('ZONE NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, VARLOCATION=([1,2]=nodal,[3,4,5,6]=CELLCENTERED), ZONETYPE=FEQUADRILATERAL\n'.format(nodeNumb,volNumb))

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
            myFile.write('{} \n'.format(h_data[i,1]))
        else:
            myFile.write('{} '.format(h_data[i,1]))
    myFile.write('\n')     

    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(h_data[i,2]))
        else:
            myFile.write('{} '.format(h_data[i,2]))
    myFile.write('\n')   

    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(h_data[i,3]))
        else:
            myFile.write('{} '.format(h_data[i,3]))
    myFile.write('\n')    

    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(h_data[i,4]))
        else:
            myFile.write('{} '.format(h_data[i,4]))
    myFile.write('\n')       
    
    
    for i in range(volNumb):
        myFile.write('{} {} {} {}\n'.format(1+volNodes[i,1],1+volNodes[i,2],1+volNodes[i,3],1+volNodes[i,4]))
    myFile.close()    
        

def createFolderHydro(tag,ER,Re,Le,Ls,volNumb,tol,Xr,dtFac,phi,Pe,alpha,Pr,Nu,beta,i):
    
    pathHydro = "./0 - Results_BFS/{}_beta={}_ER={}_Re={:.1E}_Pr={:.1E}_phi={:.1E}_Pe={:.1E}_alpha={:.2E}_Le={}_Ls={}_Volumes={}_dtFac={:.2f}_tol={}_{}k_it_Xr={:.3f}_NuDev={:.3f}".format(tag,beta,ER,Re,Pr,phi,Pe,alpha,Le,Ls,volNumb,dtFac,tol,int(i/1000),Xr,Nu)
    os.makedirs(pathHydro)
    print('')
    print('Output folder created!')    
    os.chdir(pathHydro) 
    
def createFolderHydro_PP(tag,Re,Le,h,volNumb,tol,dtFac,phi,Pe,alpha,Pr,Nu,i,ec,lamb):

    pathHydro = "./0 - Results_PP/{}/inGz={:.1E}_Ec={:.1E}_Re={:.1E}_Pr={:.1E}_phi={:.1E}_Pe={:.1E}_alpha={:.2E}_Le={}_h={}_Volumes={}_dtFac={:.2f}_tol={}_{}k_it_NuAVG={:.3f}_lambda={:.3f}".format(tag,Le/(Re*Pr*2*h),ec,Re,Pr,phi,Pe,alpha,Le,h,volNumb,dtFac,tol,int(i/1000),Nu,lamb)
    os.makedirs(pathHydro)
    print('')
    print('Output folder created!')    
    os.chdir(pathHydro) 

def createFolderHydro_LDC(tag,Re,volNumb,tol,dtFac,phi,Pe,alpha,Pr,Nu,i,h_data,lamb,ec):
    
    if Re == 100.0 and Pr == 30.0:
        Nu0 = 7.349
    else:
        Nu0 = 1.0
        print('No Nu_0 defined')
    
    pathHydro = "./0 - Results_LDC/{}/Re={:.1E}_Pr={:.1f}_phi={:.1E}_lambda={:.1f}_Pe={:.1E}_alpha={:.2E}_Ec={:.2E}_{}vols_dtFac={}_{}kIt_psiMin={:.4f}_Nu|Nu0={:.3f}".format(tag,Re,Pr,phi,lamb,Pe,alpha,ec,volNumb,dtFac,int(i/1000),np.min(h_data[:volNumb,3]),Nu/Nu0)
    os.makedirs(pathHydro)
    print('')
    print('Output folder created!')    
    os.chdir(pathHydro) 

def exportData(CVdata,field,mag,thermo,volNumb,nodeNumb,volNodes,nodesCoord,H0,nu_avg):
      
    np.savez('fieldData.npz', h_data=field, t_data=thermo, m_data=mag, CVdata=CVdata, volNumb=volNumb,nodeNumb=nodeNumb,volNodes=volNodes)
    
    # np.savez('HPflow.npz', h_data=field)
    
    myFile = open("FVfieldData.dat", 'w')
    myFile.write('VARIABLES="{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}" \n'.format("x","y","u","v","psi","w","Hx","Hy","modH","M0","Mx","My","alpha","Theta","MxH"))
    myFile.write('ZONE NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, VARLOCATION=([1,2]=nodal,[3,4,5,6,7,8,9,10,11,12,13,14,15]=CELLCENTERED), ZONETYPE=FEQUADRILATERAL\n'.format(nodeNumb,volNumb))

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
            myFile.write('{} \n'.format(mag[i,3]))
        else:
            myFile.write('{} '.format(mag[i,3]))
    myFile.write('\n')  

    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(mag[i,4]))
        else:
            myFile.write('{} '.format(mag[i,4]))
    myFile.write('\n')      
    
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(H0[i]))
        else:
            myFile.write('{} '.format(H0[i]))
    myFile.write('\n')        
    
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(np.sqrt(mag[i,1]**2 + mag[i,2]**2)))
        else:
            myFile.write('{} '.format(np.sqrt(mag[i,1]**2 + mag[i,2]**2)))
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
            myFile.write('{} \n'.format(mag[i,4]*mag[i,5] - mag[i,3]*mag[i,6]))
        else:
            myFile.write('{} '.format(mag[i,4]*mag[i,5] - mag[i,3]*mag[i,6]))
    myFile.write('\n')  
    
    for i in range(volNumb):
        myFile.write('{} {} {} {}\n'.format(1+volNodes[i,1],1+volNodes[i,2],1+volNodes[i,3],1+volNodes[i,4]))
    myFile.close()    
    
    ##################  Exporting thetaM file
    
    # myFileTheta =  open("heatTransf.dat", 'w')
    # myFileTheta.write('Variables = "x","thetaM","Nu_x", \n')
    # myFileTheta.write('\n')
    # for i in range(len(x_cv)):
    #     myFileTheta.write('{}\t {}\t {}\n'.format(x_cv[i],thetaM[i],nu_x[i]))
    
    # myFileTheta.write('\n')
    # myFileTheta.write('Nu_AVG = {}'.format(nu_avg))    
    # myFileTheta.close()
    
def gradChi_LDC(t_data,m_data,volNumb,volArr,CVdata):
    
    t_data[:,2] = m_data[:,1]/m_data[:,3]

    gradChi = np.zeros((volNumb,3))
    a = 0
    for i in range(volNumb):
        gradChi[i,0] = volArr[i,0]
        W = volArr[i,1]
        N = volArr[i,2]
        E = volArr[i,3]
        S = volArr[i,4]
        if W > 0 and N > 0 and S > 0 and E > 0:
            gradChi[i,1] = (t_data[E,2] - t_data[W,2])/(CVdata[E,2] - CVdata[W,2])
            gradChi[i,2] = (t_data[N,2] - t_data[S,2])/(CVdata[N,3] - CVdata[S,3])
            a += 1

    gradChi = np.nan_to_num(gradChi)

    avgGradChi_X = np.sum(gradChi[:,1]/a)
    avgGradChi_Y = np.sum(gradChi[:,2]/a)

    myFile = open('gradChi_avg.dat','w')
    myFile.write('grad_x = {:.4f} \n'.format(avgGradChi_X))
    myFile.write('grad_y = {:.4f} \n'.format(avgGradChi_Y))
    myFile.close()
    
    return gradChi,t_data

def exportDataPP(CVdata,field,mag,thermo,volNumb,nodeNumb,volNodes,nodesCoord,H0,x_cv,thetaM,nu_x,nu_avg):
      
    np.savez('fieldData.npz', h_data=field, t_data=thermo, m_data=mag, CVdata=CVdata, volNumb=volNumb,nodeNumb=nodeNumb,volNodes=volNodes)
    
    # np.savez('HPflow.npz', h_data=field)
    
    myFile = open("FVfieldData.dat", 'w')
    myFile.write('VARIABLES="{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}" \n'.format("x","y","u","v","psi","w","Hx","Hy","modH","M0","Mx","My","alpha","Theta","MxH"))
    myFile.write('ZONE NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, VARLOCATION=([1,2]=nodal,[3,4,5,6,7,8,9,10,11,12,13,14,15]=CELLCENTERED), ZONETYPE=FEQUADRILATERAL\n'.format(nodeNumb,volNumb))

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
            myFile.write('{} \n'.format(mag[i,3]))
        else:
            myFile.write('{} '.format(mag[i,3]))
    myFile.write('\n')  

    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(mag[i,4]))
        else:
            myFile.write('{} '.format(mag[i,4]))
    myFile.write('\n')      
    
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(H0[i]))
        else:
            myFile.write('{} '.format(H0[i]))
    myFile.write('\n')        
    
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(np.sqrt(mag[i,1]**2 + mag[i,2]**2)))
        else:
            myFile.write('{} '.format(np.sqrt(mag[i,1]**2 + mag[i,2]**2)))
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
            myFile.write('{} \n'.format(mag[i,4]*mag[i,5] - mag[i,3]*mag[i,6]))
        else:
            myFile.write('{} '.format(mag[i,4]*mag[i,5] - mag[i,3]*mag[i,6]))
    myFile.write('\n')  
    
    for i in range(volNumb):
        myFile.write('{} {} {} {}\n'.format(1+volNodes[i,1],1+volNodes[i,2],1+volNodes[i,3],1+volNodes[i,4]))
    myFile.close()    
    
    ##################  Exporting thetaM file
    
    myFileTheta =  open("heatTransf.dat", 'w')
    myFileTheta.write('Variables = "x","thetaM","Nu_x", \n')
    myFileTheta.write('\n')
    for i in range(len(x_cv)):
        myFileTheta.write('{}\t {}\t {}\n'.format(x_cv[i],thetaM[i],nu_x[i]))
    
    myFileTheta.write('\n')
    myFileTheta.write('Nu_AVG = {}'.format(nu_avg))    
    myFileTheta.close()
    

