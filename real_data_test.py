# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:41:07 2016

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
import movielens_parser as parse

#n = 15 #num users - initial parameter of R
#m = 10 #num items - initial parameter of R
d = 3 #latent features
p = 4 #num groups... start with this - this will be pruned/learned 

maxiter_main = 2
tries = 0
n_tries = 50
group_convergence = 3 #age groups need to reach before convergence met
existing_groups_prev = np.zeros(p)
existing_groups_prev.fill(False)
existing_groups_prev = np.ones((1,p), dtype=bool)
groups_age = 0

growth_frequency = 10 #num iterations between each group growth step
path = '../ml-100k/'
filename = 'u1.base'
delimiter = '\t'
R = parse.readR(path, filename, delimiter) #really this will be given - this is the recommendation system
n,m = R.shape
(W,L,Rsampled,Rmissed, a) = blc.sampleR(R,1)  #sample to make sparse
P = blc.createP(p,n) #create initial P - we don't know where users lie so this is arbitrary for now and will be learned
#P = np.zeros((p,n))
tolerance = 0.01 #error tolerance to be met for factorisation...

#**make update method that updates all the things each round - ie rtilde, Lambda, R, etc...
while ( (tries < n_tries) and (groups_age < group_convergence) ):
    
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
    """
    print "iteration number: ", tries
    print "P\n", P
    print "Rtilde\n", Rtilde
    print "Ut\n", Ut
    print "Vt\n", Vt
    print "lambda\n", Lambda
    print "groups age\n", groups_age
    """
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
        
#delete empty groups to clean up...
groups_used = P.sum(axis = 1)
P = P[groups_used!=0] #get rid of empty groups
p,n = P.shape 
Rtilde = Rtilde[groups_used!=0]#pxm
Lambda = Lambda[:, groups_used!=0]#mxp  
Ut = Ut[:,groups_used!=0] 
UV = np.dot(Ut.T, Vt)
n,m = R.shape
filename = "u1.test"
Rtest = parse.readRdefiniteSize(path, filename, delimiter, n, m)
Rtilde_testp, lam = blc.createRtilde(Rtest, P)
print "number of iterations: ", tries
print "P\n", P
print "Rtilde\n", Rtilde
print "Ut\n", Ut
print "Vt\n", Vt
print "UV - prediction!\n", UV
print "lambda\n", Lambda
print "groups age\n", groups_age 
print "factorisation error\n", blc.rms(Rtilde, Ut, Vt)       
print "prediction error\n", blc.rms(Rtilde_testp, Ut, Vt)

print "RECOMMENDATION!!~~~~~~~~~~~~~~~~~~~~"
user = np.random.randint(0,n)
group = np.argmax(P[:,user])#group the chosen user is in
print "user", user
print "user ratings\n", R[user,:]
print "group", group
print "group ratings:\n", UV[group,:]
rec_vector = blc.recommend(R[user,:], Ut, Vt, group)
print "recommendation vector\n",rec_vector
xV = np.dot(rec_vector, Vt)#this is the predicted UV from that user - gives the recommendation
Ru = np.expand_dims(R[user, :],0)
ferr = blc.rms(Ru, rec_vector.T, Vt)
print "factorisation group error in rec\n", ferr#**how should I desrcibe this error in a name??
  
rated = Ru!=0
"""
xV_copy = xV
if not(rated.all()):#
    xV_copy[rated] = np.nan
    recommend_item = np.nanargmax(xV)
    print "recommendation: item id\n", recommend_item
"""
#results.to_pickle('Rgroups_test_demo.pkl')
print "Done"
        
#**output items of user observed compared to corresponding items in xV    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    