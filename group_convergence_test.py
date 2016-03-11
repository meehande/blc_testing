# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:38:56 2016

@author: Deirdre Meehan
"""

import blc
import numpy as np

p=3
m=6
n=p*3
d=3

maxiter_main = 2
tries = 0
n_tries = 50
group_convergence = 5 #age groups need to reach before convergence met
existing_groups_prev = np.zeros(p)
existing_groups_prev.fill(False)
existing_groups_prev = np.ones((1,p), dtype=bool)
groups_age = 0

growth_frequency = 10 #num iterations between each group growth step
#R = blc.createR(n,m,d) #really this will be given - this is the recommendation system
"""
TEST: CREATE R FROM RTILDE TO TEST IF SAME GROUPS CONVERGE...
"""
Rtilde_test = np.zeros((p,m))

Rtilde_test[0,:] = [1,2,3,4,5,6]
Rtilde_test[1,:] = [6,6,6,6,0,0]
Rtilde_test[2,:] = [0,0,1,0,0,1]

R, Lambda, P = blc.RfromRtilde(Rtilde_test, 0.001)
#(W,L,Rsampled,Rmissed, a) = blc.sampleR(R,0.3)  #sample to make sparse
#P = blc.createP(p,n) #create initial P - we don't know where users lie so this is arbitrary for now and will be learned
#P = np.zeros((p,n))
tolerance = 0.01 #error tolerance to be met for factorisation...

#**make update method that updates all the things each round - ie rtilde, Lambda, R, etc...
while ( (tries < n_tries) and (groups_age < group_convergence) ):
    
    Rtilde, Lambda = blc.createRtilde(R,P)#avg rating per group for each item - pxm
   # Lambda = blc.createLambda(P,Rsampled)#users per group rating item m - pxm - 
    a = Lambda>0 #boolean index to only calculate items that have been rated by that group   
    Ut,Vt, memt = blc.ls_groups(Rtilde, d, tolerance, 10, Lambda, a) #factoristaion - groupwise
    #reallocate users to closest group
    for user in np.random.permutation(n):  
        blc.findP(R,Rtilde,user,P)#this updates P to put user in closest group by comparing R and Rtilde
    #for privacy, would only need that users row of R**include this!!? - this can be done locally by user - embarrassingly parallel
    if(not(tries%growth_frequency)):#update num groups and associated parameters
        P, Ut = blc.train_groups(P,Ut)#cull unused groups and then double them
        print "UPDATING"        
        #Rtilde and lambda will be updated on the beginning of the next iteration...
    print "iteration number: ", tries
    print "P\n", P
    print "Rtilde\n", Rtilde
    print "Ut\n", Ut
    print "Vt\n", Vt
    print "lambda\n", Lambda
    print "groups age\n", groups_age
    tries +=1
    #group convergence
    existing_groups = np.sum(a,0) > 0 #find which groups are empty/full
    if(existing_groups_prev.size == existing_groups.size):
        if((existing_groups_prev ^ existing_groups).any()): #if they're not the same
            groups_age = 0
            existing_groups_prev = existing_groups
           # print "UPDATING GROUPS"
    else:
        groups_age += 1
        #print "INCREASE GROUP AGE"
        
#delete empty groups to finish...
groups_used = P.sum(axis = 1)
P = P[groups_used!=0] #get rid of empty groups 
print "END RESULT"       
print "iteration number: ", tries
print "P\n", P
print "Rtilde\n", Rtilde
print "Ut\n", Ut
print "Vt\n", Vt
print "lambda\n", Lambda
print "groups age\n", groups_age     
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    