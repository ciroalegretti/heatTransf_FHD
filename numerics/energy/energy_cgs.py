#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:35:54 2021

@author: alegretti
"""
from numba import jit,prange
import numpy as np
from pre import idNeighbors

@jit(nopython=True, parallel=True)
def energy_cgs(volnumb,volarr,cv,wf,h_data,t_data,m_data,phi,ec,re,pe,pr,dt,tol=1e-9):
    
    # h_data = bc_uvPsi.bc_uvPsi(h_data,volarr,cv,edge,S)
    To = t_data[:,1].copy()
    Tn = To.copy()
    # W = h_data[:,4].copy()
    b = np.zeros((len(cv)))
    r = np.zeros((len(cv)))
    p = np.zeros((len(cv)))
    q = np.zeros((len(cv)))
    u = np.zeros((len(cv)))
    v = np.zeros((len(cv)))
    Au = np.zeros((len(cv)))
    Aq = np.zeros((len(cv)))
    
    for i in prange(volnumb):
        W,N,E,S = idNeighbors.idNeighbors(volarr,i)
        b[i] = Tn[i]/dt + (9/4)*t_data[i,3]*phi*(ec/(re*pe))*( \
                                                  (m_data[i,5]*m_data[i,4] - m_data[i,6]*m_data[i,3])*(h_data[E,2] - h_data[W,2])/(cv[E,2] - cv[W,2]) + \
                                                  (m_data[i,5]*m_data[i,4] - m_data[i,6]*m_data[i,3])*(h_data[N,1] - h_data[S,1])/(cv[N,3] - cv[S,3]) \
                                                      )
    # b = - h_data[:,4] + h_data[:,3]/dt
        
    l2_norm = 1.0
    
    it_number = 0
    it_max = len(cv)
    
    for i in prange(volnumb):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        W,N,E,S = idNeighbors.idNeighbors(volarr,i)
        
        r[i] = b[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                            - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                            + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
                            + (1/(re*pr*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(re*pr*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*To[i] \
                    - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*pr*dxV*(cv[E,2] - cv[i,2])))*To[E]  \
                    - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*pr*dxV*(cv[i,2] - cv[W,2])))*To[W]  \
                    - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*pr*dyV*(cv[N,3] - cv[i,3])))*To[N]  \
                    - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*pr*dyV*(cv[i,3] - cv[S,3])))*To[S]                      
                            
    r0 = r.copy()
    rho = 1#np.sum(r0*r)
    
    while it_number < it_max and l2_norm > tol:
        
        tk = To.copy()
        rk = r.copy()
        pk = p.copy()
        qk = q.copy()
        rhok = rho

        rho = np.sum(r0*r)
        beta = rho/rhok
        u = r + beta*qk
        p = u + beta*(qk + beta*pk)
        
        for i in prange(volnumb):
            dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
            dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
            W,N,E,S = idNeighbors.idNeighbors(volarr,i)
            
            v[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                            - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                            + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
                            + (1/(re*pr*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(re*pr*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*p[i] \
                    + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*pr*dxV*(cv[E,2] - cv[i,2])))*p[E]  \
                    + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*pr*dxV*(cv[i,2] - cv[W,2])))*p[W]  \
                    + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*pr*dyV*(cv[N,3] - cv[i,3])))*p[N]  \
                    + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*pr*dyV*(cv[i,3] - cv[S,3])))*p[S]                      
                            
        sigma = np.sum(r0*v)
        alpha = rho/sigma
        q = u - alpha*v
        
        for i in prange(volnumb):
            dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
            dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
            W,N,E,S = idNeighbors.idNeighbors(volarr,i)
            
            Au[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                            - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                            + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
                            + (1/(re*pr*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(re*pr*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*u[i] \
                    + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*pr*dxV*(cv[E,2] - cv[i,2])))*u[E]  \
                    + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*pr*dxV*(cv[i,2] - cv[W,2])))*u[W]  \
                    + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*pr*dyV*(cv[N,3] - cv[i,3])))*u[N]  \
                    + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*pr*dyV*(cv[i,3] - cv[S,3])))*u[S]                      
                                                       
           
            Aq[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                            - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                            + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
                            + (1/(re*pr*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(re*pr*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*q[i] \
                    + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*pr*dxV*(cv[E,2] - cv[i,2])))*q[E]  \
                    + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*pr*dxV*(cv[i,2] - cv[W,2])))*q[W]  \
                    + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*pr*dyV*(cv[N,3] - cv[i,3])))*q[N]  \
                    + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*pr*dyV*(cv[i,3] - cv[S,3])))*q[S]                      
                            
        
        r = rk - alpha*(Au + Aq)
        To = tk + alpha*(u + q)
        
        l2_norm = np.sqrt(np.sum((r)**2))
        
        it_number += 1
        
        # print(l2_norm)
    
    t_data[:,1] = To
        
    
    return t_data

@jit(nopython=True)#, parallel=True)
def energy_cgs2(volnumb,volarr,cv,wf,h_data,t_data,m_data,fv,ec,re,pe,pr,dt,tol=1e-6):
    
# =============================================================================
#     
# =============================================================================
        
    """Passo 1 - Inicializar vetores"""
    phi = t_data[:,1].copy()
    phiOld = t_data[:,1].copy()
    Q = np.zeros((len(cv)))
    r0 = np.zeros((len(cv)))
    rk = np.zeros((len(cv)))
    Gk = np.zeros((len(cv)))
    v = np.zeros((len(cv)))
    A_dC = np.zeros((len(cv)))
    A_G = np.zeros((len(cv)))
    
    # myfile = open('./0 - Results_CGSMaz/Ex{}_R2_{}vols_tol{:.1E}.dat'.format(MAZ,volnumb,tol),'w')
    # myfile.write('Variables="it","R2"\n')
    
    """Termo independente do S.L."""
    """Maz 3.2"""
    for i in prange(volnumb):
                 
        W,N,E,S = idNeighbors.idNeighbors(volarr,i)
        Q[i] = phiOld[i]/dt + (9/4)*t_data[i,3]*fv*(ec/(re*pe))*( \
                                                   (m_data[i,5]*m_data[i,4] - m_data[i,6]*m_data[i,3])*(h_data[E,2] - h_data[W,2])/(cv[E,2] - cv[W,2]) + \
                                                   (m_data[i,5]*m_data[i,4] - m_data[i,6]*m_data[i,3])*(h_data[N,1] - h_data[S,1])/(cv[N,3] - cv[S,3]) \
                                                       )

    R2 = 1.0
    it_number = 0
    it_max = volnumb

    """Passo 2 - Inicializar resíduo"""
    for i in prange(volnumb):
        dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
        dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
        W,N,E,S = idNeighbors.idNeighbors(volarr,i)

        r0[i] = Q[i] - (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                            - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                            + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
                            + (1/(re*pr*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(re*pr*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*phiOld[i] \
                    - ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*pr*dxV*(cv[E,2] - cv[i,2])))*phiOld[E]  \
                    - ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*pr*dxV*(cv[i,2] - cv[W,2])))*phiOld[W]  \
                    - ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*pr*dyV*(cv[N,3] - cv[i,3])))*phiOld[N]  \
                    - ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*pr*dyV*(cv[i,3] - cv[S,3])))*phiOld[S]                         

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
    
            v[i] = + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                            - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                            + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
                            + (1/(re*pr*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(re*pr*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*dOld[i] \
                    + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*pr*dxV*(cv[E,2] - cv[i,2])))*dOld[E]  \
                    + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*pr*dxV*(cv[i,2] - cv[W,2])))*dOld[W]  \
                    + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*pr*dyV*(cv[N,3] - cv[i,3])))*dOld[N]  \
                    + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*pr*dyV*(cv[i,3] - cv[S,3])))*dOld[S]    
    
        alphak = np.sum(r0*rOld)/np.sum(r0*v)
    
        """Passo 5 - atualizar o operador Gk"""
        Gk = dC - alphak*v
    
        for i in prange(volnumb):
            dxV = cv[i,4]#nodesCoord[trn,1] - nodesCoord[tln,1]
            dyV = cv[i,5]#nodesCoord[trn,2] - nodesCoord[brn,2]
            W,N,E,S = idNeighbors.idNeighbors(volarr,i)
            
            A_dC[i] =    + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                            - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                            + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
                            + (1/(re*pr*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(re*pr*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*dC[i] \
                    + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*pr*dxV*(cv[E,2] - cv[i,2])))*dC[E]  \
                    + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*pr*dxV*(cv[i,2] - cv[W,2])))*dC[W]  \
                    + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*pr*dyV*(cv[N,3] - cv[i,3])))*dC[N]  \
                    + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*pr*dyV*(cv[i,3] - cv[S,3])))*dC[S]                              
           
            A_G[i] =     + (1/dt + (wf[i,1]/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1])  \
                            - (wf[i,3]/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1])  \
                            + (wf[i,2]/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2])  \
                            - (wf[i,4]/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2])  \
                            + (1/(re*pr*dxV))*(1/(cv[E,2] - cv[i,2]) + 1/(cv[i,2] - cv[W,2]))     \
                            + (1/(re*pr*dyV))*(1/(cv[N,3] - cv[i,3]) + 1/(cv[i,3] - cv[S,3])))*Gk[i] \
                    + ( + ((1 - wf[i,1])/dxV)*(wf[i,1]*h_data[i,1] + (1 - wf[i,1])*h_data[E,1]) - 1/(re*pr*dxV*(cv[E,2] - cv[i,2])))*Gk[E]  \
                    + ( - ((1 - wf[i,3])/dxV)*(wf[i,3]*h_data[i,1] + (1 - wf[i,3])*h_data[W,1]) - 1/(re*pr*dxV*(cv[i,2] - cv[W,2])))*Gk[W]  \
                    + ( + ((1 - wf[i,2])/dyV)*(wf[i,2]*h_data[i,2] + (1 - wf[i,2])*h_data[N,2]) - 1/(re*pr*dyV*(cv[N,3] - cv[i,3])))*Gk[N]  \
                    + ( - ((1 - wf[i,4])/dyV)*(wf[i,4]*h_data[i,2] + (1 - wf[i,4])*h_data[S,2]) - 1/(re*pr*dyV*(cv[i,3] - cv[S,3])))*Gk[S]   
                         
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
        
        # print(it_number,R2)
        # myfile.write('{} \t {:.5E} \n'.format(it_number,R2))
        
        it_number += 1
        
    t_data[:,1] = phi
    print(it_number)
    # myfile.close()
    
    return t_data