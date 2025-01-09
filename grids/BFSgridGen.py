"""
Created on Wed Apr 25 18:25:19 2018
@author: alegretti
"""
import gridGenFunc
import numpy as np
import time

t0 = time.time()
"""############################################################################

                                INPUTS
                                
############################################################################"""
"""Geometrical parameters"""
ER = 2.0   # ER = 1.9423
beta = 1.0
# betaArr = [1.0,2.0,3.0,4.0]   # stretching grid paramater
#betaArr = np.flip(betaArr)
# n = 1

h = 0.5
H = ER*h
S = H - h
Le = 10*h  # Lenght of entrance channel

for n in [3.5,4.0,4.5,5]:#[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5]:#,6,7,8,9,10]:
    Ls = 20*h*n  # Lenght of expansion channel
    #    for beta in betaArr:
    """Numerical parameters"""
    nx1 = 100#int(50*n)#int(45*1.1**(2*n))
    ny1 = 20#50#int(20*n)#int(30*1.1**(2*n))
    ##################################################
    nx2 = int(200*n)#500#int(120*n)#3*nx1
    ny2 = ny1
    nx3 = nx2
    ny3 = 20#int(20*n)#50#int(1.25*ny1)       # doimain 3 with 50% more nodes in y-direction than domain 2 for ER=2 and proportional when ER=1.5 and 3.0
    print('nx1 = {}, ny1 = {}, nx2 = {}, ny3 = {}'.format(nx1,ny1,nx2,ny3))
    ##################################################
    nodeNumb,volNumb = 0,0                                      # starting counters
    volNodes = np.array([[0., 0., 0., 0., 0.]]) 
    volArr = np.array([[0., 0., 0., 0., 0.]])
    
    print('Initializing FVM grid generation')
    print('ER = {}, beta = {}, Le = {}, Ls = {}, nx1 = {}, ny1 = ny2 = {}, nx2 = nx3 = {}, ny3 = {}'.format(ER,beta,Le,Ls,nx1,ny1,nx2,ny3))
    """############################################################################
    
                                  GRID CONSTRUCTION
                                 
    ############################################################################"""
    print('1/7 - Calculating nodes coordinates and steps for each subdomain')
    x1,y1,x2,y2,x3,y3 = gridGenFunc.starters(nx1,ny1,nx2,ny2,nx3,ny3,H,h,S,Le,Ls)
    """Finding the node on the step edge"""
    e = gridGenFunc.idStepNode(nx1,nx2,ny1)
    """Grid generation"""
    x1,y1 = gridGenFunc.allocateXY(x1,y1)
    x2,y2 = gridGenFunc.allocateXY(x2,y2)
    x3,y3 = gridGenFunc.allocateXY(x3,y3) 
    """Normalizing before streching"""
    x1,x2,x3 = gridGenFunc.normalizeB4strech(x1,x2,x3,Ls,Le)
    """stretching grid in x-direction"""
    x1 = gridGenFunc.strechX(1.0,x1,Le,Le,-1,d=0.01*Le)
    x2,x3 = gridGenFunc.strechX(beta,x2,Ls,Le,1,d=0.01*Ls), gridGenFunc.strechX(beta,x3,Ls,Le,1,d=0.01*Ls)
    """Removing matching nodes"""
    x1,y1,x3,y3,nx1,ny3 = gridGenFunc.rm_coincidentNodes(x1,y1,x3,y3,nx1,ny3)
    print('Done')
    
    """Lists creation"""
    print('2/7 - Generating nodes coordinates array')
    nodesCoord,nodeNumb = gridGenFunc.XYcoordArray(x1,x2,x3,y1,y2,y3,nodeNumb)
    print('Done')
    
    print('3/7 - Generating volumes coordinates array')
    volNodes,volNumb = gridGenFunc.defineVolumes(nx1+nx2,ny1  ,0,volNumb,volNodes)
    volNodes = volNodes[1:]     # Clean array starter
    volNodes,volNumb = gridGenFunc.defineVolumes(nx3    ,ny3+1,e,volNumb,volNodes)
    print('Done')
    
    print('4/7 - Creating conectivity matrix')
    volArr = gridGenFunc.volumesArrangement(volNumb,volNodes,volArr)
    print('Done')
    
    print('5/7 - Inserting ghosts volumes')
    volArr,edgeVol = gridGenFunc.insertGhosts(volNumb,volNodes,volArr,e,nx2)
    print('Done')
    
    print('6/7 - Calculating volume areas and wall functions interpolations')
    CVdata = gridGenFunc.calcAreas(volArr,volNodes,nodesCoord,edgeVol)
    wallFunc = gridGenFunc.cell2faceInterp(volArr,volNodes,nodesCoord,CVdata)
    print('Done')
    
    print('7/7 - Exporting data')
    """Exporting grid data"""
    gridGenFunc.exportData(CVdata,wallFunc,volNumb,volNodes,volArr,nodesCoord,nodeNumb,nx1,ny1,nx2,ny2,nx3,ny3,x1,x2,x3,y1,y2,y3,e,edgeVol,Le,Ls,S,ER,beta)
    print('Done')
    
    print("Run time (h): {}".format((time.time() - t0)/3600))
    print("")
    print("Current grid volumes count: {}".format(volNumb))