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

def createFolderHydro_LDC(tag,Re,volNumb,tol,dtFac,phi,Pe,alpha,Pr,Nu,i,h_data,lamb,beta_dt):
    
    if Re == 100.0 and Pr == 30.0:
        if volNumb == 6561:
            Nu0 = 7.349                  #grad T vert
            # Nu0 = 10.0563                    #grad T horiz 
        elif volNumb == 10201:
            Nu0 = 7.3609                 #grad T vert
            # Nu0 = 10.0799                  #grad T horiz 
    else:
        Nu0 = 1.0
        print('No Nu_0 defined')
            
    # pathHydro = "./0 - Results_LDC/{}/Re={:.1E}_Pr={:.1f}_phi={:.1E}_lambda={:.1f}_Pe={:.1E}_alpha={:.2E}_Ec={:.2E}_{}vols_dtFac={}_{}kIt_psiMin={:.4f}_Nu0={:.4f}".format(tag,Re,Pr,phi,lamb,Pe,alpha,ec,volNumb,dtFac,int(i/1000),np.min(h_data[:volNumb,3]),Nu)
    pathHydro = "./0 - Results_LDC/{}/Re={:.1E}_Pr={:.1f}_phi={:.1E}_lambda={:.1f}_Pe={:.1E}_alpha={:.1E}_betaDt={:.1f}_{}vols_dtFac={}_{}kIt_psiMin={:.4f}_Nu|Nu0={:.4f}".format(tag,Re,Pr,phi,lamb,Pe,alpha,beta_dt,volNumb,dtFac,int(i/1000),np.min(h_data[:volNumb,3]),Nu/Nu0)
    os.makedirs(pathHydro)
    print('')
    print('Output folder created!')    
    os.chdir(pathHydro) 

def exportData(CVdata,field,mag,thermo,volNumb,nodeNumb,volNodes,nodesCoord,H0,nu_avg,tm,lap):
      
    np.savez('fieldData.npz', h_data=field, t_data=thermo, m_data=mag, CVdata=CVdata, volNumb=volNumb,nodeNumb=nodeNumb,volNodes=volNodes)
    
    # np.savez('HPflow.npz', h_data=field)
    
    myFile = open("FVfieldData.dat", 'w')
    myFile.write('VARIABLES="x","y","u","v","psi","w","Hx","Hy","modH","M0","Mx","My","alpha","Theta","chi","lap(MxH)","tm_torque" \n')
    myFile.write('ZONE NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, VARLOCATION=([1,2]=nodal,[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]=CELLCENTERED), ZONETYPE=FEQUADRILATERAL\n'.format(nodeNumb,volNumb))

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
            myFile.write('{} \n'.format(thermo[i,-2]))
        else:
            myFile.write('{} '.format(thermo[i,-2]))
    myFile.write('\n')       
    
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(thermo[i,1]))
        else:
            myFile.write('{} '.format(thermo[i,1]))
    myFile.write('\n')  
    
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(thermo[i,4]))
        else:
            myFile.write('{} '.format(thermo[i,4]))
    myFile.write('\n')  
    
    
    # -lap(MxH)
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(lap[i,1]))
        else:
            myFile.write('{} '.format(lap[i,1]))
    myFile.write('\n')  
        
                   
    ### MxH
    # for i in range(volNumb):
    #     if i % 1000 == 0:
    #         myFile.write('{} \n'.format(mag[i,4]*mag[i,5] - mag[i,3]*mag[i,6]))
    #     else:
    #         myFile.write('{} '.format(mag[i,4]*mag[i,5] - mag[i,3]*mag[i,6]))
    # myFile.write('\n')  
    
    ###
    for i in range(volNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(tm[i,1]))
        else:
            myFile.write('{} '.format(tm[i,1]))
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
    
    t_data[:,-1] = np.sqrt(m_data[:,1]**2 + m_data[:,2]**2)/np.sqrt(m_data[:,3]**2 + m_data[:,4]**2)   #M0/H
    # t_data[:,2] = np.sqrt(m_data[:,-1]**2 + m_data[:,-2]**2)/np.sqrt(m_data[:,3]**2 + m_data[:,4]**2)   # Full Mag M

    gradChi = np.zeros((volNumb,3))
    a = 0
    for i in range(volNumb):
        gradChi[i,0] = volArr[i,0]
        W = volArr[i,1]
        N = volArr[i,2]
        E = volArr[i,3]
        S = volArr[i,4]
        if W > 0 and N > 0 and S > 0 and E > 0:
            gradChi[i,1] = (t_data[E,-1] - t_data[W,-1])/(CVdata[E,2] - CVdata[W,2])
            gradChi[i,2] = (t_data[N,-1] - t_data[S,-1])/(CVdata[N,3] - CVdata[S,3])
            a += 1

    gradChi = np.nan_to_num(gradChi)

    avgGradChi_X = np.sum(gradChi[:,1])/a
    avgGradChi_Y = np.sum(gradChi[:,2])/a

    myFile = open('gradChi_avg.dat','w')
    myFile.write('grad_x = {:.4f} \n'.format(avgGradChi_X))
    myFile.write('grad_y = {:.4f} \n'.format(avgGradChi_Y))
    myFile.close()
    
    return gradChi,t_data


def calculate_tmTorque(volNumb,volArr,t_data,CVdata,modH):
    
    tm_torque = np.zeros((volNumb,2))
    a = 0
    for i in range(volNumb):
        tm_torque[i,0] = volArr[i,0]
        W = volArr[i,1]
        N = volArr[i,2]
        E = volArr[i,3]
        S = volArr[i,4]
        if W > 0 and N > 0 and S > 0 and E > 0:
            tm_torque[i,1] = (((t_data[E,-1] - t_data[W,-1])/(CVdata[E,2] - CVdata[W,2]))*((modH[N]**2/2 - modH[S]**2/2)/(CVdata[N,3] - CVdata[S,3]))) \
                           - (((t_data[N,-1] - t_data[S,-1])/(CVdata[N,3] - CVdata[S,3]))*((modH[E]**2/2 - modH[W]**2/2)/(CVdata[E,2] - CVdata[W,2])))
        a += 1

    # tm_torque = np.nan_to_num(tm_torque)
    
    # Smoothing ghost volumes
    for i in range(volNumb):
        W = volArr[i,1]
        N = volArr[i,2]
        E = volArr[i,3]
        S = volArr[i,4]
        if W < 0:
            tm_torque[W,1] = (5*tm_torque[i,1] - tm_torque[E,1])/4
        if E < 0:
            tm_torque[E,1] = (5*tm_torque[i,1] - tm_torque[W,1])/4
        if N < 0:
            tm_torque[N,1] = (5*tm_torque[i,1] - tm_torque[S,1])/4
        if S < 0:
            tm_torque[S,1] = (5*tm_torque[i,1] - tm_torque[N,1])/4
    
    return tm_torque
    

def calc_lap_MxH(volNumb,volArr,mag,CVdata):
    
    # -Laplacian(MxH)
    
    lap = np.zeros((volNumb,2))
    a = 0
    for i in range(volNumb):
        dxV = CVdata[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = CVdata[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        lap[i,0] = volArr[i,0]
        W = volArr[i,1]
        N = volArr[i,2]
        E = volArr[i,3]
        S = volArr[i,4]
        if W > 0 and N > 0 and S > 0 and E > 0:    
            lap[i,1] = \
                    + ((1/(dxV))*(1/(CVdata[E,2] - CVdata[i,2]) + 1/(CVdata[i,2] - CVdata[W,2]))     \
                     + (1/(dyV))*(1/(CVdata[N,3] - CVdata[i,3]) + 1/(CVdata[i,3] - CVdata[S,3])))*(mag[i,4]*mag[i,5] - mag[i,3]*mag[i,6]) \
                    - ( 1/(dxV*(CVdata[E,2] - CVdata[i,2])))*(mag[E,4]*mag[E,5] - mag[E,3]*mag[E,6])  \
                    - ( 1/(dxV*(CVdata[i,2] - CVdata[W,2])))*(mag[W,4]*mag[W,5] - mag[W,3]*mag[W,6])  \
                    - ( 1/(dyV*(CVdata[N,3] - CVdata[i,3])))*(mag[N,4]*mag[N,5] - mag[N,3]*mag[N,6])  \
                    - ( 1/(dyV*(CVdata[i,3] - CVdata[S,3])))*(mag[S,4]*mag[S,5] - mag[S,3]*mag[S,6]) 
        a += 1
        
                    
    return lap


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
    

