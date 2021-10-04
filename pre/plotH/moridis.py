#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 01:35:25 2021

@author: alegretti
"""
import numpy as np

def moridis(x,y,z,Hx,Hy,Hz,a,b,Br):
    """
    
    Solving Clegg equations for H field for a permanent magnet positioned externaly
    at the step face (no field before sudden expansion)
       
    magData = [#vol, M0x, M0y, Hx, Hy, Mx, My]    
               
    """
    
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                
                Hx[i,j,k] = (Br)*(np.arctan(((y[i,j,k]+a)*(z[i,j,k]+b))/(x[i,j,k]*(((y[i,j,k]+a)**2+(z[i,j,k]+b)**2+x[i,j,k]**2)**(0.5)))) \
                          + np.arctan(((y[i,j,k]-a)*(z[i,j,k]-b))/(x[i,j,k]*(((y[i,j,k]-a)**2+(z[i,j,k]-b)**2+x[i,j,k]**2)**(0.5))))\
                          - np.arctan(((y[i,j,k]+a)*(z[i,j,k]-b))/(x[i,j,k]*(((y[i,j,k]+a)**2+(z[i,j,k]-b)**2+x[i,j,k]**2)**(0.5))))\
                          - np.arctan(((y[i,j,k]-a)*(z[i,j,k]+b))/(x[i,j,k]*(((y[i,j,k]-a)**2+(z[i,j,k]+b)**2+x[i,j,k]**2)**(0.5)))))
                
                Hy[i,j,k] = (Br)*(np.log((((z[i,j,k]+b)+((z[i,j,k]+b)**2+(y[i,j,k]-a)**2+x[i,j,k]**2)**(0.5))/((z[i,j,k]-b)+\
                            ((z[i,j,k]-b)**2+(y[i,j,k]-a)**2+x[i,j,k]**2)**(0.5)))*(((z[i,j,k]-b)+((z[i,j,k]-b)**2+(y[i,j,k]+a)**2\
                             +x[i,j,k]**2)**(0.5))/((z[i,j,k]+b)+((z[i,j,k]+b)**2+(y[i,j,k]+a)**2+x[i,j,k]**2)**(0.5)))))
                
                Hz[i,j,k] = (Br)*(np.log((((y[i,j,k]+a)+((z[i,j,k]-b)**2+(y[i,j,k]+a)**2+x[i,j,k]**2)**(0.5))/((y[i,j,k]-a)+\
                            ((z[i,j,k]-b)**2+(y[i,j,k]-a)**2+x[i,j,k]**2)**(0.5)))*(((y[i,j,k]-a)+((z[i,j,k]+b)**2+(y[i,j,k]-a)**2\
                             +x[i,j,k]**2)**(0.5))/((y[i,j,k]+a)+((z[i,j,k]+b)**2+(y[i,j,k]+a)**2+x[i,j,k]**2)**(0.5)))))

                    
    return Hx,Hy,Hz  

nx = 50
ny = 50
nz = 50

x = np.linspace(-1,1,nx)
y = np.linspace(-1,1,ny)
z = np.linspace(-1,1,nz)

X,Y,Z = np.meshgrid(x,y,z)

HxN,HyN,HzN = X.copy(),Y.copy(),Z.copy()

HxS = HxN.copy()
HyS = HyN.copy()
HzS = HzN.copy()

J = 1
b = 0.1#S/2    # Clegg magnet height = 2b
a = 0.1#1E+5     # Clegg magnet width = 2a
Lm = 0.1

HxN,HyN,HzN = moridis(X, Y, Z, HxN, HyN, HzN,a,b,J)                                                                                                         
HxS,HyS,HzS = moridis(X+Lm, Y, Z, HxS, HyS, HzS,a,b,J)     

Hx = HxN - HxS
Hy = HyN - HyS
Hz = HzN - HzS
                                                                               

myFile = open("permMag.dat", 'w')
myFile.write('Variables="{}","{}","{}","{}","{}","{}", \n'.format("X","Y",\
          "Z","Hx","Hy","Hz"))
myFile.write('ZONE F=POINT,I={},J={},K={} \n'.format(nx,ny,nz))
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            myFile.write('\t{}\t{}\t{}\t{}\t{}\t{} \n'.\
            format(X[i,j,k],Y[i,j,k],Z[i,j,k],Hx[i,j,k],Hy[i,j,k],Hz[i,j,k]))
myFile.close()  

# myFile = open("permMag2D.dat", 'w')
# myFile.write('Variables="{}","{}","{}","{}", \n'.format("X","Y",\
#          "Hx","Hy"))
# myFile.write('ZONE F=POINT,I={},J={} \n'.format(nx,ny))
# for i in range(nx):
#     for j in range(ny):
#         myFile.write('\t{}\t{}\t{}\t{} \n'.\
#         format(X[i,j,0],Y[i,j,0],Hx[i,j,0],Hy[i,j,0]))
# myFile.close()  