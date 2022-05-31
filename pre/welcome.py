#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 16:58:06 2022

@author: alegretti
"""
import sys
import os
from pre import loadGrid

def inputModel():

    label = input("Enter a label for the output folder: ")
    
    print("Available cases: \n \n (1) for Lid Driven Cavity (LDC) \n (2) for Parallel Plates (PP) \n (3) for Backward Facing Step (BFS)")
    model = input("Enter model: ")
    
    return model,label

def welcome(model):
    
    
    # print('\n')
    # print('\n')
    # LDC
    if model == '1':
        Le = 1.0
        h = 1.0
        path_to_grid_LDC = './grids/LDC/'
        gridsLDC = sorted(os.listdir(path_to_grid_LDC),reverse=True)
        vols = input('{} \n \n Select grid (N x N) file entering the number of volumes: \n \n'.format(gridsLDC))
        
        grid_check = os.path.exists('./grids/LDC/gridData_{}_volumes_L={}_h={}.npz'.format(vols,Le,h))
        if grid_check == True:
            nx1,ny1,volArr,volNodes,nodesCoord,x1,y1,CVdata,volNumb,nodeNumb,Le,h,wallFunc = loadGrid.loadGrid_LDC(vols,Le,h)

            return nx1,ny1,volArr,volNodes,nodesCoord,x1,y1,CVdata,volNumb,nodeNumb,Le,h,wallFunc

        else:
            print('\n no Grid available \n')
            sys.exit()        

            
    elif model == '2':
        
        h=0.5
        Le = 10.0#float(input('Enter dimensionless lenght of the paralell plates channel (h=0.5): '))
        
        path_to_grid_PP = './grids/PP'
        gridsPP = sorted(os.listdir(path_to_grid_PP),reverse=True)
        vols = input('{} \n \n Select grid file entering the number of volumes: \n \n'.format(gridsPP))
        
        grid_check = os.path.exists('./grids/PP/gridData_{}_volumes_Le={}_h={}.npz'.format(vols,Le,h))
        if grid_check == True:
            nx1,ny1,volArr,volNodes,nodesCoord,x1,y1,CVdata,volNumb,nodeNumb,Le,h,wallFunc = loadGrid.loadGrid_PP(vols,Le,h)

            return nx1,ny1,volArr,volNodes,nodesCoord,x1,y1,CVdata,volNumb,nodeNumb,Le,h,wallFunc

        else:
            print('\n no Grid available \n')
            sys.exit()     
    
    elif model =='3':
        ER = 2.0            # Expansion ratio
        h = 0.5             # Channel height before expansion
        Le = 5.0            # Channel Lenght before expansion
        Ls = 15.0#float(input('Select the lenght of the channel at the expanded section: '))
        beta = float(input('Select beta grid refinement: '))
        
        path_to_grid_BFS = './grids/BFS/ER=2.0/beta={}/'.format(beta)
        gridsBFS = sorted(os.listdir(path_to_grid_BFS),reverse=True)
        vols = input('{} \n \n Select grid file entering the number of volumes: \n \n'.format(gridsBFS))
        
        gridDataCheck = os.path.exists("./grids/BFS/ER={}/beta={}/gridData_{}_volumes_Le={}_Ls={}_ER={}.npz".format(ER,beta,vols,Le,Ls,ER))
        
        if gridDataCheck == True:
            nx1,nx2,nx3,ny1,ny2,ny3,volArr,volNodes,nodesCoord,x1,x2,x3,y1,y2,y3,e,edgeVol,CVdata,volNumb,nodeNumb,Le,Ls,S,wallFunc = loadGrid.loadGrid_BFS(vols,Le,Ls,ER,beta)
            
            return nx1,nx2,nx3,ny1,ny2,ny3,volArr,volNodes,nodesCoord,x1,x2,x3,y1,y2,y3,e,edgeVol,CVdata,volNumb,nodeNumb,Le,Ls,S,wallFunc,ER,beta
        
        else:
            print('\n no Grid available \n')
            sys.exit()        
        
    else:
        sys.exit()


