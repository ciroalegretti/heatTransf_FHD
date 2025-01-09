"""
Created on Wed Apr 25 18:25:19 2018
@author: alegretti
"""
import PPgridGenFunc
import numpy as np
import time

t0 = time.time()
"""############################################################################

                                INPUTS
                                
############################################################################"""
"""Geometrical parameters"""
# invGz = 0.01                                         # Gz^-1 = L/(2*h*Re*Pr)
L = 10                                              # Channel Lenght
h = 0.5                                             # So Dh = 1.0
Pr = 100                                            # Prandtl number
# ReArr = np.logspace(-1,0.,2)
# Re = L/(2*h*Pr*invGz)
dxArr = [0.01,]#[0.2,0.1,0.05,0.025]

# LeArr = ReArr.copy()
# for k in range(len(dxArr)):
    # dx = dxArr[k]
for n in [1,2,3,4,5,6,7,8]:
# for Le in LeArr:
    """Numerical parameters"""
    nx1 = 201#n*50 + 1#int(1 + L/dx)
    ny1 = n*10 + 1 #81#50#int(20*n)#int(30*1.1**(2*n))
    ##################################################
    print('nx1 = {}, ny1 = {}'.format(nx1,ny1))
    ##################################################
    nodeNumb,volNumb = 0,0                                      # starting counters
    volNodes = np.array([[0., 0., 0., 0., 0.]]) 
    volArr = np.array([[0., 0., 0., 0., 0.]])
    
    print('Initializing FVM grid generation')
    """############################################################################
    
                                  GRID CONSTRUCTION
                                 
    ############################################################################"""
    print('1/7 - Calculating nodes coordinates and steps for each subdomain')
    x1,y1 = PPgridGenFunc.starters(nx1,ny1,h,L)
    """Grid generation"""
    x1,y1 = PPgridGenFunc.allocateXY(x1,y1)
    print('Done')
    
    """Lists creation"""
    print('2/7 - Generating nodes coordinates array')
    nodesCoord,nodeNumb = PPgridGenFunc.XYcoordArray(x1,y1,nodeNumb)
    print('Done')
    
    print('3/7 - Generating volumes coordinates array')
    volNodes,volNumb = PPgridGenFunc.defineVolumes(nx1,ny1,0,volNumb,volNodes)
    volNodes = volNodes[1:]     # Clean array starter
    print('Done')
    
    print('4/7 - Creating conectivity matrix')
    volArr = PPgridGenFunc.volumesArrangement(volNumb,volNodes,volArr)
    volArr = volArr.astype(int)
    print('Done')
    
    print('5/7 - Inserting ghosts volumes')
    volArr = PPgridGenFunc.insertGhosts(volNumb,volNodes,volArr,)
    print('Done')
    
    print('6/7 - Calculating volume areas and wall functions interpolations')
    CVdata = PPgridGenFunc.calcAreas(volArr,volNodes,nodesCoord)
    wallFunc = PPgridGenFunc.cell2faceInterp(volArr,volNodes,nodesCoord,CVdata)
    print('Done')
    
    print('7/7 - Exporting data')
    """Exporting grid data"""
    PPgridGenFunc.exportData(CVdata,wallFunc,volNumb,volNodes,volArr,nodesCoord,nodeNumb,nx1,ny1,x1,y1,L,h)
    print('Done')
    
    print("Run time (h): {}".format((time.time() - t0)/3600))
    print("")
    print("Current grid volumes count: {}".format(volNumb))