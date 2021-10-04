"""
Created on Thu May 10 20:00:27 2018

@author: alegretti

Grid generation and mapping functions for the Finite Volumes Method applied to 
the 2-D laminar flow over a Backward-facing step:

y axis ^             h = H - S = entrance height
  y01 _| 
       |#####################---------------------------
       |#####################---------------------------
       |#######   1   #######-----------  2  -----------
       |#####################---------------------------
  y03 _|#####################---------------------------
       |                     @@@@@@@@@@@@@@@@@@@@@@@@@@@     ----> Flow
       |                     @@@@@@@@@@@@@@@@@@@@@@@@@@@
       |                     @@@@@@@@@@@  3  @@@@@@@@@@@
       |                     @@@@@@@@@@@@@@@@@@@@@@@@@@@
       |                     @@@@@@@@@@@@@@@@@@@@@@@@@@@
       |_______________________________________________________> x axis 
       0 = x01               |                         
                            x02   

obs.: in order to guarantee matching nodes between domains: 
    ny1 = ny2, y01 = y02 and dy1 = dy2, 
    nx2 = nx3, x02 = x03 and dx2 = dx3,
    
______________________________________________________________________________________    
                                    PARAMETERS
______________________________________________________________________________________
    NAME                 DESCRIPTION                              DIMENSION, TYPE          
_______________________________________________________________________________________                                                                    
nx1, nx2:   nodes in x-direction of each domain                   [1,int]
ny1, ny3:   nodes in y-direction of each domain                   [1,int]
x01, x02:   domains 1 and 2/3 x-coordinates starting points
            (node distribution from left to right)                [1,float]
dx1, dx2:   grid steps in x-direction                             [1,float]
y01, y03:   domains 1/2 and 3 y-coordinates starting points       
            (as nodes distribution starts from the top)           [1,float]
dy1, dy3:   grid steps in y-direction                             [1,float]
nodeNumb:   node count                                            [1,int]
nodesCoord: 2D array for node number and its xy coordinates.
            Format:  
            [[nodeNumber, xCoord, yCoord]]                        [(nodeNumb,3), float64]
volNumb:    volume count                                          [1,int]
volNodes:   2D array for mapping the nodes of each volume
            Format:
            [[volNumb,topleft,topright,bottomright,bottomleft]]   [(nodeNumb,5), int64]
volArr:     2D array mapping neighbor volumes
            Format:
            [[volNumb, left, top, right, bottom]]                 [(nodeNumb,5), int64]  
_______________________________________________________________________________________

"""
import numpy as np
from numba import jit

@jit(nopython=True)
def normalizeB4strech(x1,x2,x3,Ls,Le):

    x1 = x1/Le
    x2 = (x2 - Le)/(Ls - Le)
    x3 = (x3 - Le)/(Ls - Le)    
    
    return x1,x2,x3

def rm_coincidentNodes(x1,y1,x3,y3,nx1,ny3):
    x1 = np.flip(x1)   # to flip finest grid location to the right
    x1 = x1[:,:-1]
    y1 = y1[:,:-1]
    x1[:,0] = 0.0   # Forcing  "x0 = 0.0"
    
    y3 = y3[1:,:]
    x3 = x3[1:,:]
    y3[-1,:] = 0.0  # Forcing  "y0 = 0.0"
    
    nx1 -= 1
    ny3 -= 1
    
    return x1,y1,x3,y3,nx1,ny3

@jit(nopython=True)
def strechX(beta,xold,l,c,b,d):

    # d =  0.1*l
    
    A = np.log( (1 + (np.exp(beta)-1)*(d/l))/(1 + (np.exp(-beta)-1)*(d/l)))/(2*beta)
    
    xnew = xold.copy()
    for k in range(len(xnew)):
        
        xnew[k] = c + b*d*(1. + np.sinh(beta*(xold[k] - A))/np.sinh(beta*A))
    
    return xnew

@jit(nopython=True)
def starters(nx1,ny1,nx2,ny2,nx3,ny3,H,h,S,Le,Ls):
    """
    Creates nodes coordinates arrays for each domain
    """
    x1 = np.linspace(0,Le,nx1)
    y1 = np.linspace(H,S,ny1)
    
    x2 = np.linspace(Le,Ls,nx2)
    y2 = y1
    
    x3 = x2
    y3 = np.linspace(S,0,ny3)
    
    return x1,y1,x2,y2,x3,y3

def idStepNode(nx1,nx2,ny1):
    return (nx1+nx2 - 1)*ny1 - nx2

# @jit(nopython=True)
def allocateXY(x,y):
    """
    Allocate x and y coordinates to its 2d-arrays for equivalent domain
    before interpolation.
    
        x : empty x-coordinates matrix             [2d-array]
        x0: domain x-coordinate starting point     [float]
        y : empty x-coordinates matrix             [2d-array]
        y0: domain y-coordinate starting point     [float]
        dx: grid step in x-direction               [float]
        dy: grid step in y-direction               [float]        
        
    """
    x,y = np.meshgrid(x,y)



    # for j in range(ny):
    #     for i in range(nx):
    #         x[:,i] = x0 + i*dx
    #         y[j,:] = y0 - j*dy
            
    return x, y

def XYcoordArray(x1,x2,x3,y1,y2,y3,nodeNumb):
    """
    Creates an array with shape (nodeNumb,3), specifying:
        
        [nodeNumber, x coordinate, y coordinate]
    
    """
    ny1,nx1 = x1.shape
    ny2,nx2 = x2.shape
    ny3,nx3 = y3.shape  
    nodesCoord = np.array([[0., 0., 0.]])
    for j in range(ny1):
        for i in range(nx1):
            nodesCoord = np.append(nodesCoord, np.array([[nodeNumb, x1[j,i], y1[j,i]]]), axis = 0)
            nodeNumb +=1  
        for k in range(nx2):
            nodesCoord = np.append(nodesCoord, np.array([[nodeNumb, x2[j,k], y2[j,k]]]), axis = 0)    
            nodeNumb +=1     
    for j in range(ny3):
        for i in range(nx3):
            nodesCoord = np.append(nodesCoord, np.array([[nodeNumb, x3[j,i], y3[j,i]]]), axis = 0)
            nodeNumb +=1         
    
    nodesCoord = nodesCoord[1:]
            
    return nodesCoord, nodeNumb    

def defineVolumes(nx,ny,e,volNumb,volNodes):
    """
    Creates an array with shape (volNumb,5), specifying:
        
        [volumeNumber, top-left node, top-right node, bottom-right node, bottom-left node]
    
    """    
    for j in range(ny-1):
        for i in range(nx-1):
            volNodes = np.append(volNodes, np.array([[volNumb, \
                                                      e + i + j*nx,\
                                                      e + i + j*nx + 1, \
                                                      e + i + (j+1)*nx + 1, \
                                                      e + i + (j+1)*nx ]]), axis = 0)

            volNumb += 1
    
    volNodes = volNodes.astype(int)
        
    return volNodes,volNumb

def volumesArrangement(volNumb,volNodes,volArr):       
    """
    Conectivity matriz:
        
        volArr[] = [volumeNumber, left volume, top volume, right volume, bottom volume]
        
        OBS.: if neighbor number = -1, there is no neighbor volume in such direction (volumes adjacent to
        domain boundaries) and a ghost volume will be later placed
    
    """    
    for j in range(volNumb):
        a = volNodes[j,1]
        b = volNodes[j,2]
        c = volNodes[j,3]
        d = volNodes[j,4]
        L,R,B,T = -1,-1,-1,-1
        for i in range(volNumb):
            if volNodes[i,2] == a and volNodes[i,3] == d:
                L = volNodes[i,0]
            if volNodes[i,1] == b and volNodes[i,4] == c:
                R = volNodes[i,0]
            if volNodes[i,1] == d and volNodes[i,2] == c:
                B = volNodes[i,0]
            if volNodes[i,3] == b and volNodes[i,4] == a:
                T = volNodes[i,0]
                
        volArr = np.append(volArr, np.array([[j, L, T, R, B ]]), axis = 0)
    volArr = volArr[1:]  
    volArr = volArr.astype(int)

    return volArr

@jit(nopython=True)
def idNeighbors(volArr,N):
    """
    Recieves a volume number and returns the numbers of its four neighbors
    """
    w = volArr[N,1]
    n = volArr[N,2]
    e = volArr[N,3]
    s = volArr[N,4]
    
    return w,n,e,s

@jit(nopython=True)
def insertGhosts(volNumb,volNodes,volArr,e,nx2):
    """
    Maps ghosts volumes ordering arround domain boundaries, starting at the top of volume #0 to clockwise orientation back at the left of volume #0
    
    Attributes negative values for ghosts volumes surrounding real volumes on the volumes arrangement table "volArr"
    
    """    
    ghost = -1
    """Top volumes"""
    for k in range(volNumb):
        left,top,right,bottom = idNeighbors(volArr,k)        
        if top == -1:
            volArr[k,2] = ghost
            ghost -= 1
    """Right volumes"""        
    for k in range(volNumb):
        left,top,right,bottom = idNeighbors(volArr,k)
        if right == -1:
            volArr[k,3] = ghost
            ghost -= 1
    """ Finding volume which has bottom-left node as "e" """        
    for k in range(volNumb):
        if volNodes[k,4] == e:
            edgeVol = k            
    """bottom volumes at the expansion domain"""
    for k in range(volNumb,edgeVol,-1):
        left,top,right,bottom = idNeighbors(volArr,k)        
        if bottom == -1:
            volArr[k,4] = ghost
            ghost -= 1     
    """left volumes at the step face"""            
    for k in range(volNumb,edgeVol,-1):
        left,top,right,bottom = idNeighbors(volArr,k)        
        if left == -1:
            volArr[k,1] = ghost
            ghost -= 1   
    # """Manually setting the matching ghost volume for the two real volumes located at the top and the right at the step edge"""
#    volArr[edgeVol-1,4] = volArr[edgeVol + nx2 - 1, 1]
    """bottom volumes at the entrance domain"""
    for k in range(edgeVol-1,0,-1):
        left,top,right,bottom = idNeighbors(volArr,k)        
        if bottom == -1:
            volArr[k,4] = ghost
            ghost -= 1       
    """left volumes at the entrance channel left boundary"""
    for k in range(edgeVol-1,-1,-1):
        left,top,right,bottom = idNeighbors(volArr,k)        
        if left == -1:
            volArr[k,1] = ghost
            ghost -= 1   
            
    return volArr, edgeVol

@jit(nopython=True)
def calcAreas(volArr,volNodes,nodesCoord,edgeVol):
    """
    Calculate the area and centroid coordinates of each control volume, including ghosts
    
    controlVol = [volNumber, area, xC, yC, dxV, dyV]
    
    """
    n = len(volArr)
    lastGhost = np.min(volArr) 
    gi = np.abs(lastGhost) # last ghost abs index
    controlVol = np.zeros((n + gi,6))
    
    for k in range(n,n+gi):
        controlVol[k,0] = lastGhost
        lastGhost += 1
        
    for k in range(n):
        tln = volNodes[k,1]
        trn = volNodes[k,2]
        brn = volNodes[k,3]
        bln = volNodes[k,4]
        controlVol[k,0] = k
        controlVol[k,1] = (nodesCoord[tln,2] - nodesCoord[bln,2])*(nodesCoord[brn,1] - nodesCoord[bln,1])
        controlVol[k,2] = (nodesCoord[brn,1] + nodesCoord[bln,1])/2.
        controlVol[k,3] = (nodesCoord[tln,2] + nodesCoord[bln,2])/2.  
        controlVol[k,4] = (nodesCoord[brn,1] - nodesCoord[bln,1])
        controlVol[k,5] = (nodesCoord[tln,2] - nodesCoord[bln,2])
        """Appending data for associated ghosts volumes"""
        W,N,E,S = idNeighbors(volArr,k)
        if W < 0:
            controlVol[W,1] = controlVol[k,1] # area
            tln = volNodes[k,1]
            controlVol[W,2] = 2.*nodesCoord[tln,1] - controlVol[k,2]
            controlVol[W,3] = controlVol[k,3] # y coordinate
            controlVol[W,4] = controlVol[k,4]
            controlVol[W,5] = controlVol[k,5]
        if N < 0:
            controlVol[N,1] = controlVol[k,1]
            controlVol[N,2] = controlVol[k,2]
            tln = volNodes[k,1]
            controlVol[N,3] = 2.*nodesCoord[tln,2] - controlVol[k,3]
            controlVol[N,4] = controlVol[k,4]
            controlVol[N,5] = controlVol[k,5]
        if E < 0:
            controlVol[E,1] = controlVol[k,1]
            trn = volNodes[k,2]
            controlVol[E,2] = 2.*nodesCoord[trn,1] - controlVol[k,2]
            controlVol[E,3] = controlVol[k,3]
            controlVol[E,4] = controlVol[k,4]
            controlVol[E,5] = controlVol[k,5]
        if S < 0:
            controlVol[S,1] = controlVol[k,1]
            controlVol[S,2] = controlVol[k,2]
            bln = volNodes[k,4]
            controlVol[S,3] = 2.*nodesCoord[bln,2] - controlVol[k,3]
            controlVol[S,4] = controlVol[k,4]
            controlVol[S,5] = controlVol[k,5]

#    """Imposing correct values for the ghost volume on the step corner"""    
#    mixedAreaGhostVolume = volArr[edgeVol - 1,4]
#    controlVol[mixedAreaGhostVolume,1] = controlVol[edgeVol,1]
#    controlVol[mixedAreaGhostVolume,2] = controlVol[edgeVol - 1,2]
#    SedgeVol = volArr[edgeVol,4]
#    controlVol[mixedAreaGhostVolume,3] = controlVol[SedgeVol,3]
    
#    """Overwriting correct values for ghosts on the step faces in respect to the corner volume coordinates"""
#    for k in range(edgeVol,n):
#        W = volArr[k,1]
#        if W < 0:
#            controlVol[W,2] = controlVol[mixedAreaGhostVolume,2]
#    for k in range(0,edgeVol):
#        S = volArr[k,4]
#        if S < 0:
#            controlVol[S,3] = controlVol[mixedAreaGhostVolume,3]
        
    return controlVol

def exportData(CVdata,wallFunc,volNumb,volNodes,volArr,nodesCoord,nodeNumb,nx1,ny1,nx2,ny2,nx3,ny3,x1,x2,x3,y1,y2,y3,e,edgeVol,Le,Ls,S,ER,beta):
    """
    Export grid data to both .dat (with tecplot360 compatible headers) and .npz formats
    """    
    np.savez('./ER={}/beta={}/gridData_{}_volumes_Le={}_Ls={}_ER={}.npz'.format(ER,beta,volNumb,Le,Ls,ER), nx1=nx1, nx2=nx2, nx3=nx3, ny1=ny1, ny2=ny2, ny3=ny3, volArr=volArr, wallFunc = wallFunc, volNodes=volNodes, nodesCoord=nodesCoord, \
                             x1=x1, x2=x2, x3=x3, y1=y1, y2=y2, y3=y3, e=e, edgeVol=edgeVol, CVdata=CVdata, volNumb=volNumb, nodeNumb=nodeNumb,Le=Le,Ls=Ls,S=S)
    
    myFile = open("./ER={}/beta={}/FVgrid_{}_volumes_Le={}_Ls={}_ER={}.dat".format(ER,beta,volNumb,Le,Ls,ER), 'w')
    myFile.write('VARIABLES="{}","{}" \n'.format("x","y"))
    myFile.write('ZONE NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, ZONETYPE=FEQUADRILATERAL\n'.format(nodeNumb,volNumb))

    for i in range(nodeNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(nodesCoord[i,1]))
        else:
            myFile.write('{} \n'.format(nodesCoord[i,1]))
    myFile.write('\n')    

    for i in range(nodeNumb):
        if i % 1000 == 0:
            myFile.write('{} \n'.format(nodesCoord[i,2]))
        else:
            myFile.write('{} '.format(nodesCoord[i,2]))
    myFile.write('\n')      

    # for i in range(volNumb):
    #     myFile.write('{} {} {} {}\n'.format(1+volNodes[i,1],1+volNodes[i,2],1+volNodes[i,3],1+volNodes[i,4]))
    # myFile.close()
    
    for i in range(volNumb):
        myFile.write('{} {} {} {}\n'.format(1+volNodes[i,1],1+volNodes[i,2],1+volNodes[i,3],1+volNodes[i,4]))
    myFile.close() 
    
    print('.npz and .dat data files exported!')      
    print("")



def cell2faceInterp(volArr,volNodes,nodesCoord,cv):
    
    n = len(volArr)
    wf =  np.zeros((n,5))
    
    """ Wall function - quadratic interpolation of fluxes through volumes of different sizes [Mazumder, 2015]
    
    wf = [volNumber, left frontier, top frontier, right frontier, bottom frontier] 
    """    
    for k in range(n):
        tln = volNodes[k,1]
        # trn = volNodes[k,2]
        brn = volNodes[k,3]
        # bln = volNodes[k,4]
        
        W,N,E,S = idNeighbors(volArr,k)
        
        # volNumb
        wf[k,0] = k
        
        # East function dx-interp
        wf[k,1] = (1/(cv[E,2] - nodesCoord[brn,1]))/(1/(cv[E,2] - nodesCoord[brn,1]) + \
                                                     1/(nodesCoord[brn,1] - cv[k,2]))        
        # North function dy-interp
        wf[k,2] = (1/(cv[N,3] - nodesCoord[tln,2]))/(1/(cv[N,3] - nodesCoord[tln,2]) + \
                                                     1/(nodesCoord[tln,2] - cv[k,3]))
        # West function dx-interp
        wf[k,3] = (1/(nodesCoord[tln,1] - cv[W,2]))/(1/(nodesCoord[tln,1] - cv[W,2]) + \
                                                     1/(cv[k,2] - nodesCoord[tln,1]))
        # South function dy-interp
        wf[k,4] = (1/(nodesCoord[brn,2] - cv[S,3]))/(1/(nodesCoord[brn,2] - cv[S,3]) + \
                                                     1/(cv[k,3] - nodesCoord[brn,2]))
       
    return wf