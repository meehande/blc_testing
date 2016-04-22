# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 10:50:48 2016

@author: Deirdre Meehan
"""
try:
    xrange
except NameError:
    xrange = range
try: input = raw_input
except NameError: pass

import blc
import numpy as np
import movielens_parser as parse
import timeit

#-------INITIALISE VALUES-----------------------------------------#
d = 3 #latent features
p = 4 #num groups... start with this - this will be pruned/learned 
tries = 0
n_tries = 50
group_convergence = 3 #age groups need to reach before convergence met
#existing_groups_prev = np.zeros(p)
#existing_groups_prev.fill(False)
existing_groups_prev = np.ones((1,p), dtype=bool)
groups_age = 0
growth_frequency = 10 #num iterations between each group growth step
tolerance = 0.01 #error tolerance to be met for factorisation...
path = '../ml-100k/'
filename = 'u1.base'
delimiter = '\t'
"""
u1.base:
float64 - Rsampled = 12710016 bytes
float16 - Rsampled = 3177504 bytes

"""

#------------------READ DATA-------------------------------------#
start_time = timeit.default_timer()
Rsampled = parse.readR(path, filename, delimiter) #really this will be given - this is the recommendation system
read_time = timeit.default_timer() - start_time
n,m = Rsampled.shape
#create random initial P - we don't know where users lie so this is arbitrary for now and will be learned
P = blc.createP(p,n) 

#------------------LEARNING--------------------------------------#
start_loop_time = timeit.default_timer()
while ( (tries < n_tries) and (groups_age < group_convergence) ):
    
    Rtilde, Lambda = blc.createRtilde(Rsampled,P)#avg rating per group for each item - pxm; number of users behind each rating
    a = Lambda>0 #boolean index to only calculate items that have been rated by that group   
    factor_stime = timeit.default_timer()    
    Ut,Vt, memt = blc.ls_groups(Rtilde, d, tolerance, 10, Lambda, a) #factoristaion - groupwise
    factor_time = timeit.default_timer() - factor_stime    
    #--------------REALLOCATE USERS------------------------------#
    for user in np.random.permutation(n):  
        blc.findP(Rsampled,Rtilde,user,P)#this updates P to put user in closest group by comparing R and Rtilde
    
    if(not(tries%growth_frequency)):#update num groups and associated parameters
        P, Ut = blc.train_groups(P,Ut)#cull unused groups and then double them
        #print "UPDATING"        
    #--------------GROUP CONVERGENCE-----------------------------#
    existing_groups = np.sum(a,0) > 0 #find which groups are empty/full
    if(existing_groups_prev.size == existing_groups.size):
        if((existing_groups_prev ^ existing_groups).any()): #if they're not the same
            groups_age = 0
            existing_groups_prev = existing_groups
    else:
        groups_age += 1
        
    tries +=1
    
learn_time = timeit.default_timer() - start_loop_time   
#------------CLEAN UP------------------------------------------------------------------#
groups_used = P.sum(axis = 1)
P = P[groups_used!=0] #get rid of empty groups
p,n = P.shape 
Rtilde = Rtilde[groups_used!=0]#pxm
Lambda = Lambda[:, groups_used!=0]#mxp  
Ut = Ut[:,groups_used!=0] 
#predicted Rtilde
UV = np.dot(Ut.T, Vt)
#prediction error
n,m = Rsampled.shape
filename = "u1.test"
Rtest = parse.readRdefiniteSize(path, filename, delimiter, n, m)
Rtilde_test, lam = blc.createRtilde(Rtest, P)
clean_time = timeit.default_timer() - learn_time
tot_time = timeit.default_timer() - start_time

#------------RESULTS-------------------------------------------------------------------#
print "\nRESULTS!"
print "Number of iterations: ", tries
print "\nUt: (dxp)\n", Ut
print "\nV: (dxm)\n", Vt
print "\nLambda: (mxp)\n", Lambda
print "\nP: (pxn)\n", P
print "\nRtilde: (pxm)\n", Rtilde
print "\nUV - prediction!: (pxm)\n", UV
print "\nGroups age:\n", groups_age 


rated_g = Rtilde!=0
rated_u = Rsampled!=0
density = float(np.sum(rated_u))/float(np.sum(rated_u==False))
print "\nDensity of data:\n", density
print "\nObserved values:\n", Rtilde[rated_g]
print "Factorised equivalent:\n", UV[rated_g]

#-----------ERROR-------------------------------------------------------------------------#
print "\nFactorisation Error:\n", blc.rms(Rtilde, Ut, Vt)       
print "Prediction Error:\n", blc.rms(Rtilde_test, Ut, Vt)
#-----------MEMORY------------------------------------------------------------------------#
print "\nR size (bytes):\n", Rsampled.nbytes
print "Rtilde size(bytes):\n", Rtilde.nbytes
#----------TIMING-------------------------------------------------------------------------#
print "\nTotal Time: (s)\n", tot_time
print "Learning Time: (s)\n", learn_time
print "Reading Time: (s)\n", read_time
#print "Done"
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    