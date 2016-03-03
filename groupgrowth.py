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
import numpy as np


"""
Group Learning alg:
Start:  need R (Rsampled) = nxm
        decide on starting value for p (num groups) and create random P based on this
Iterate:
    1. findRtilde(R,P) - group ratings matrix
    2. createLambda(P,r) - #people per group-item rating 
    3. Factorisation - find Ut,Vt
    4. findP(R,Rtilde,user,P) - for each user, update the group they are in to be the closest to their ratings
    (5.) train groups - delete unused and double remaining number - only do every few iterations to allow time for new groups to be populated 
"""

"""
TO DO:
**COLD START
**CONVERGENCE OF GROUPS & ERROR CHECKS
**PERFORMANCE MEASURES
**COMPARE TO MATLAB DIRECTLY - VERIFY IT WORKS CORRECTLY!
"""

maxiter_main = 2
tries = 0
n_tries = 50
done = False
n = 15 #num users - initial parameter of R
m = 10 #num items - initial parameter of R
d = 2 #latent features
p = 1 #num groups... start with this - this will be pruned/learned 
growth_frequency = 10 #num iterations between each group growth step
R = blc.createR(n,m,d) #really this will be given - this is the recommendation system
(W,L,Rsampled,Rmissed, a) = blc.sampleR(R,0.3)  #sample to make sparse
P = blc.createP(p,n) #create initial P - we don't know where users lie so this is arbitrary for now and will be learned
#P = np.zeros((p,n))
tolerance = 0.01 #error tolerance to be met for factorisation...

#**make update method that updates all the things each round - ie rtilde, Lambda, R, etc...
while ( (tries < n_tries) ):#and (not(done)) ):
    
    Rtilde, Lambda = blc.createRtilde(Rsampled,P)#avg rating per group for each item - pxm
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
    tries +=1
"""    
    
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
            blc.createLambda(P,R)
  """          
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    