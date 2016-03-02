# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 18:09:40 2016

@author: Deirdre Meehan
"""

# compatibility python 3
try:
    xrange
except NameError:
    xrange = range
try: input = raw_input
except NameError: pass

import blc
import timeit
import oct2py
import numpy as np
from memory_profiler import memory_usage
import pandas as pd # http://pandas.pydata.org/pandas-docs/stable/10min.html
import matplotlib # http://pandas.pydata.org/pandas-docs/stable/visualization.html
#matplotlib.style.use('ggplot')  # sudo add-apt-repository ppa:takluyver/matplotlib-daily
import pdb
import os.path
import sys
import pickle

maxiter_main = 2

tries = 0
n_tries = 5
done = False
n = 4
m = 5
d = 2
p = 3
R = blc.createR(n,m,d)
(W,L,Rsampled,Rmissed, a) = blc.sampleR(R,0.3)  # sample from it **rho
P = blc.createP(p,n)
tolerance = 0.01

#**make update method that updates all the things each round - ie rtilde, Lambda, R, etc...
while ( (tries < n_tries) and (not(done)) ):
    
    Rtilde = blc.createRtilde(Rsampled,P)#avg rating per group for each item - pxm
    Lambda = blc.createLambda(P,Rsampled)#users per group rating item m - pxm
    a = Lambda>0    
    Ut,Vt, memt = blc.ls_groups(Rtilde, d, tolerance, 10, Lambda, a)
    for iii in xrange(1,maxiter_main):
        if done:
            break
        aa = 0;
        for iiii in np.random.permutation(n):
            aa = aa+1
            blc.findP(R,Rtilde,iiii, P)#assign each user to groups closest to their ratings
            if(aa%5): #**placeholder to get things started...
                blc.train_groups(P,Ut)
            Ut,Vt, memt = blc.ls_groups(Rtilde, d, tolerance, 10, Lambda, a)
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    