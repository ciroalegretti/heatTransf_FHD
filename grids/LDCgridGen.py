"""
Created on Wed Apr 25 18:25:19 2018
@author: alegretti
"""
import LDCgridGenFunc
import numpy as np

"""############################################################################

                                INPUTS
                                
############################################################################"""
"""Geometrical parameters"""
h = 1.0  # Cavity width
L = 1.0  # Cavity height

"""Numerical parameters"""
for nx1 in [162,]:#[42,82,162,322]:
    nx1 = int(nx1)#321#10*n + 1#int(L/0.02) + 1#1000#int(50*n)#int(45*1.1**(2*n))
    ny1 = nx1#10*n + 1#50#int(20*n)#int(30*1.1**(2*n))
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
    x1,y1 = LDCgridGenFunc.starters(nx1,ny1,h,L)
    """Grid generation"""
    x1,y1 = LDCgridGenFunc.allocateXY(x1,y1)
    print('Done')
    
    """Lists creation"""
    print('2/7 - Generating nodes coordinates array')
    nodesCoord,nodeNumb = LDCgridGenFunc.XYcoordArray(x1,y1,nodeNumb)
    print('Done')
    
    print('3/7 - Generating volumes coordinates array')
    volNodes,volNumb = LDCgridGenFunc.defineVolumes(nx1,ny1,0,volNumb,volNodes)
    volNodes = volNodes[1:]     # Clean array starter
    print('Done')
    
    print('4/7 - Creating conectivity matrix')
    volArr = LDCgridGenFunc.volumesArrangement(volNumb,volNodes,volArr)
    volArr = volArr.astype(int)
    print('Done')
    
    print('5/7 - Inserting ghosts volumes')
    volArr = LDCgridGenFunc.insertGhosts(volNumb,volNodes,volArr,)
    print('Done')
    
    print('6/7 - Calculating volume areas and wall functions interpolations')
    CVdata = LDCgridGenFunc.calcAreas(volArr,volNodes,nodesCoord)
    wallFunc = LDCgridGenFunc.cell2faceInterp(volArr,volNodes,nodesCoord,CVdata)
    print('Done')
    
    print('7/7 - Exporting data')
    """Exporting grid data"""
    LDCgridGenFunc.exportData(CVdata,wallFunc,volNumb,volNodes,volArr,nodesCoord,nodeNumb,nx1,ny1,x1,y1,L,h)
    print('Done')
    
    print("Grid generated successfully")
    print("")
    print("Current grid volumes count: {}".format(volNumb))