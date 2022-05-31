#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:22:28 2021

@author: alegretti
"""
import numpy as np

def finalFields(h_data,t_data,ER,Re,Pr,Le,Ls,volNumb):
    
    np.savez('./t0/t0_ER={}_Re={:.1E}_Pr={:.1E}_Le={}_Ls={}_Volumes={}'.format(ER,Re,Pr,Le,Ls,volNumb), h_data = h_data, t_data = t_data)

