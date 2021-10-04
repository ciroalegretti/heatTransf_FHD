#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:10:43 2021

@author: alegretti
"""
import time

def endPrint(Re,Xr,step,nu_avg,t0):
    
    print('')    
    print("Literature results: ")
    print("G.Biswas: Xr/S(Re=50) = 1.6795")
    print("E.Erturk: Xr/S(Re=100) = 2.878")
    print("E.Erturk: Xr/S(Re=200) = 4.9")
    print("E.Erturk: Xr/S(Re=300) = 6.659")
    print("E.Erturk: Xr/S(Re=400) = 8.178")
    print('')
    print("A. Nada (2008): Nu_avg(Re=200) = 2.15")
    print('')
    print("Current simulation @Re = {}".format(Re))
    print("Xr/S = {:.2f}".format(Xr/step))
    print("Nu_avg = {:.2f}".format(nu_avg))
    print('')
    print("Run time (h): {}".format((time.time() - t0)/3600))
    print('')
    print('')
    print('')