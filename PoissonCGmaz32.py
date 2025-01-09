#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:10:58 2022

@author: alegretti
"""
from pre import loadGrid
from pre import idNeighbors
import numpy as np
from numba import jit,prange
import time           

# @jit(nopython=True)#, parallel=True)
def poisson_cgsSonn(MAZ,volnumb,volarr,cv,h_data,tol):
    
# =============================================================================
#     
# =============================================================================
        
    """Passo 1 - Inicializar vetores"""
    phi = np.zeros((len(cv)))#h_data[:,1].copy()
    Q = np.zeros((len(cv)))
    r0 = np.zeros((len(cv)))
    rk = np.zeros((len(cv)))
    Gk = np.zeros((len(cv)))
    v = np.zeros((len(cv)))
    A_dC = np.zeros((len(cv)))
    A_G = np.zeros((len(cv)))
    
    myfile = open('./0 - Results_CGSMaz/Ex{}_R2_{}vols_tol{:.1E}.dat'.format(MAZ,volnumb,tol),'w')
    myfile.write('Variables="it","R2"\n')
    
    """Aplicando CC"""
    # phi = bc_phiMaz(phi,volArr,cv)
    for i in prange(volnumb):
        W,N,E,S = idNeighbors.idNeighbors(volArr,i)
        
        if MAZ == 1:
            """Maz 3.1"""
            #"""Left wall"""
            if W < 0:
                phi[W] = 1000*(((CVdata[W,2] - 1/2)**2)*np.sinh((CVdata[W,2] - 1/2)) + ((CVdata[W,3] - 1/2)**2)*np.sinh((CVdata[W,3] - 1/2)))
    
            #"""Right wall"""
            if E < 0:
                phi[E] = 1000*(((CVdata[E,2] - 1/2)**2)*np.sinh((CVdata[E,2] - 1/2)) + ((CVdata[E,3] - 1/2)**2)*np.sinh((CVdata[E,3] - 1/2)))
    
            #""" Bottom wall"""
            if S < 0:    
                phi[S] = 1000*(((CVdata[S,2] - 1/2)**2)*np.sinh((CVdata[S,2] - 1/2)) + ((CVdata[S,3] - 1/2)**2)*np.sinh((CVdata[S,3] - 1/2)))
    
            #"""top"""
            if N < 0:    
                phi[N] = 1000*(((CVdata[N,2] - 1/2)**2)*np.sinh((CVdata[N,2] - 1/2)) + ((CVdata[N,3] - 1/2)**2)*np.sinh((CVdata[N,3] - 1/2)))
            
        elif MAZ == 2:
            """Maz 3.2"""
            #"""Left wall"""
            if W < 0:
                phi[W] = ((CVdata[W,2] - 1/2)**2)*np.sinh(10*(CVdata[W,2] - 1/2)) + ((CVdata[W,3] - 1/2)**2)*np.sinh(10*(CVdata[W,3] - 1/2)) + np.exp(2*CVdata[W,2]*CVdata[W,3])
    
            #"""Right wall"""
            if E < 0:
                phi[E] = ((CVdata[E,2] - 1/2)**2)*np.sinh(10*(CVdata[E,2] - 1/2)) + ((CVdata[E,3] - 1/2)**2)*np.sinh(10*(CVdata[E,3] - 1/2)) + np.exp(2*CVdata[E,2]*CVdata[E,3])
    
            #""" Bottom wall"""
            if S < 0:    
                phi[S] = ((CVdata[S,2] - 1/2)**2)*np.sinh(10*(CVdata[S,2] - 1/2)) + ((CVdata[S,3] - 1/2)**2)*np.sinh(10*(CVdata[S,3] - 1/2)) + np.exp(2*CVdata[S,2]*CVdata[S,3])
    
            #"""top"""
            if N < 0:    
                phi[N] = ((CVdata[N,2] - 1/2)**2)*np.sinh(10*(CVdata[N,2] - 1/2)) + ((CVdata[N,3] - 1/2)**2)*np.sinh(10*(CVdata[N,3] - 1/2)) + np.exp(2*CVdata[N,2]*CVdata[N,3])
            
    """Termo independente do S.L."""
    for i in prange(volnumb):
        if MAZ == 1:
            """Maz 3.1"""
            Q[i] = + 1000*(2*np.sinh(cv[i,2] - 1/2) + 4*(cv[i,2] - 1/2)*np.cosh(cv[i,2] - 1/2) + ((cv[i,2] - 1/2)**2)*np.sinh(cv[i,2] - 1/2)) \
                   + 1000*(2*np.sinh(cv[i,3] - 1/2) + 4*(cv[i,3] - 1/2)*np.cosh(cv[i,3] - 1/2) + ((cv[i,3] - 1/2)**2)*np.sinh(cv[i,3] - 1/2))
        if MAZ == 2:
            """Maz 3.2"""
            Q[i] =   (2*np.sinh(10*(cv[i,2] - 1/2)) + 40*(cv[i,2] - 1/2)*np.cosh(10*(cv[i,2] - 1/2)) + 100*((cv[i,2] - 1/2)**2)*np.sinh(10*(cv[i,2] - 1/2)))\
                   + (2*np.sinh(10*(cv[i,3] - 1/2)) + 40*(cv[i,3] - 1/2)*np.cosh(10*(cv[i,3] - 1/2)) + 100*((cv[i,3] - 1/2)**2)*np.sinh(10*(cv[i,3] - 1/2)))\
                   + 4*(cv[i,2]**2 + cv[i,3]**2)*np.exp(2*cv[i,2]*cv[i,3])

    R2 = 1.0
    it_number = 0
    it_max = volnumb

    """Passo 2 - Inicializar resíduo"""
    for i in prange(volnumb):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        W,N,E,S = idNeighbors.idNeighbors(volarr,i)

        r0[i] = Q[i] - ( - (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                         - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*phi[i] \
                    - ( + 1/(dxV*(cv[E,2] - cv[i,2])))*phi[E]  \
                    - ( + 1/(dxV*(cv[i,2] - cv[W,2])))*phi[W]  \
                    - ( + 1/(dyV*(cv[N,3] - cv[i,3])))*phi[N]  \
                    - ( + 1/(dyV*(cv[i,3] - cv[S,3])))*phi[S]                         

    """Passo 3 - Inicializar vetores direção e direção conjugada"""
    d = r0
    dC = r0
    rk = r0

    # while it_number < 2:
    while it_number < it_max and R2 > tol:

        rOld = rk.copy()
        dOld = d.copy()     

        """Passo 4 - atualizar o avanço alpha^{n+1}"""
        for i in prange(volnumb):
            dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
            dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
            W,N,E,S = idNeighbors.idNeighbors(volarr,i)
    
            v[i] =   ( - (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                       - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*dOld[i] \
                   + ( + 1/(dxV*(cv[E,2] - cv[i,2])))*dOld[E]  \
                   + ( + 1/(dxV*(cv[i,2] - cv[W,2])))*dOld[W]  \
                   + ( + 1/(dyV*(cv[N,3] - cv[i,3])))*dOld[N]  \
                   + ( + 1/(dyV*(cv[i,3] - cv[S,3])))*dOld[S]   
    
        alphak = np.sum(r0*rOld)/np.sum(r0*v)
    
        """Passo 5 - atualizar o operador Gk"""
        Gk = dC - alphak*v
    
        for i in prange(volnumb):
            dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
            dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
            W,N,E,S = idNeighbors.idNeighbors(volarr,i)
            
            A_dC[i] =       ( - (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                              - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*dC[i] \
                          + ( + 1/(dxV*(cv[E,2] - cv[i,2])))*dC[E]  \
                          + ( + 1/(dxV*(cv[i,2] - cv[W,2])))*dC[W]  \
                          + ( + 1/(dyV*(cv[N,3] - cv[i,3])))*dC[N]  \
                          + ( + 1/(dyV*(cv[i,3] - cv[S,3])))*dC[S]                              
           
            A_G[i] =       ( - (1/(dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                             - (1/(dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*Gk[i] \
                         + ( + 1/(dxV*(cv[E,2] - cv[i,2])))*Gk[E]  \
                         + ( + 1/(dxV*(cv[i,2] - cv[W,2])))*Gk[W]  \
                         + ( + 1/(dyV*(cv[N,3] - cv[i,3])))*Gk[N]  \
                         + ( + 1/(dyV*(cv[i,3] - cv[S,3])))*Gk[S] 
                         
        """Passo 6 - atualizar estimativa de phi"""
        phi += alphak*(dC + Gk)    
    
        """Passo 7 - Atualizar o resíduo e R2"""
        rk = rOld - alphak*(A_dC + A_G)  # Sonnenveld
        
        R2 = np.sqrt(np.sum(rk*rk))
        
        """Passo 8 - Calcular coef. de combinação"""
        beta = np.sum(r0*rk)/np.sum(r0*rOld)    
        
        """Passo 9 - Atualizar vetor direção conjugada"""
        dC = rk + beta*Gk        
        
        """Passo 10 - Atualizar vetor direção de busca"""
        d = dC + beta*(Gk + beta*dOld)   
        
        print(it_number,R2)
        myfile.write('{} \t {:.5E} \n'.format(it_number,R2))
        
        it_number += 1
        
    
    myfile.close()
    
    return phi,it_number,Q

"""                   Selected volNumb: 400,1600,3600,6400,10000                           """
tol = 1E-6
MAZ = 2

volsArr = [400,1600,6400,25600]#,40000,90000,160000,250000,360000]

for volNumb in volsArr:
    
    t0 = time.time()

    volNumb = int(volNumb)
    
    nx1,ny1,volArr,volNodes,nodesCoord,x1,y1,CVdata,volNumb,nodeNumb,L,h,wallFunc = loadGrid.loadGrid_LDC(volNumb,1.0,1.0)
    
    phi_data = CVdata[:,:3].copy()

    """Solving Poisson equation for phi"""
    phi_data[:,1],it,Q = poisson_cgsSonn(MAZ,volNumb,volArr,CVdata,phi_data,tol)
    """Analytical solution"""
    for i in range(volNumb):
        if MAZ == 1:
            """Maz 3.1"""
            phi_data[i,2] = 1000*(((CVdata[i,2] - 1/2)**2)*np.sinh((CVdata[i,2] - 1/2)) + ((CVdata[i,3] - 1/2)**2)*np.sinh((CVdata[i,3] - 1/2)))
        if MAZ == 2:
            """Maz 3.2"""
            phi_data[i,2] = ((CVdata[i,2] - 1/2)**2)*np.sinh(10*(CVdata[i,2] - 1/2))\
                          + ((CVdata[i,3] - 1/2)**2)*np.sinh(10*(CVdata[i,3] - 1/2))\
                          + np.exp(2*CVdata[i,2]*CVdata[i,3])

    print('vols = {} \t Iter. = {} \t Erro máximo % = {:.5f} \t RunTime(s) = {:.2f}'.format(volNumb,it,100*np.max(np.abs(phi_data[:volNumb,1] - phi_data[:volNumb,2]))/np.max(np.abs(phi_data[:volNumb,2])),(time.time() - t0)))

    phiOut = open('./0 - Results_CGSMaz/Ex{}_sol_{}vols_tol{:.1E}.dat'.format(MAZ,volNumb,tol),'w')
    phiOut.write('VARIABLES="x","y","phiGCS","phiAn","errPhi" \n')
    phiOut.write('ZONE NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, VARLOCATION=([1,2]=nodal,[3,4,5]=CELLCENTERED), ZONETYPE=FEQUADRILATERAL\n'.format(nodeNumb,volNumb))

    # Write x
    for i in range(nodeNumb):
        if i % 1000 == 0:
            phiOut.write('{} \n'.format(nodesCoord[i,1]))
        else:
            phiOut.write('{} '.format(nodesCoord[i,1]))
    phiOut.write('\n')    

    # Write y
    for i in range(nodeNumb):
        if i % 1000 == 0:
            phiOut.write('{} \n'.format(nodesCoord[i,2]))
        else:
            phiOut.write('{} '.format(nodesCoord[i,2]))
    phiOut.write('\n')    

    # Write phiCGS
    for i in range(volNumb):
        if i % 1000 == 0:
            phiOut.write('{} \n'.format(phi_data[i,1]))
        else:
            phiOut.write('{} '.format(phi_data[i,1]))
    phiOut.write('\n')

    # Write phiAn
    for i in range(volNumb):
        if i % 1000 == 0:
            phiOut.write('{} \n'.format(phi_data[i,2]))
        else:
            phiOut.write('{} '.format(phi_data[i,2]))
    phiOut.write('\n')

    # Write phiErr
    for i in range(volNumb):
        if i % 1000 == 0:
            phiOut.write('{} \n'.format(np.abs(phi_data[i,2]-phi_data[i,1])))
        else:
            phiOut.write('{} '.format(phi_data[i,2]-phi_data[i,1]))
    phiOut.write('\n')
    for i in range(volNumb):
        phiOut.write('{} {} {} {}\n'.format(1+volNodes[i,1],1+volNodes[i,2],1+volNodes[i,3],1+volNodes[i,4]))

    phiOut.close()    
    
