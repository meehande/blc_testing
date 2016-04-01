# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 10:17:17 2016

@author: Deirdre Meehan
"""
import blc
import numpy as np
"""
METHOD TO BE CALLED FROM GUI
dataset to be specified:
- real data 
- 
"""
m=6
n=9
R_groups = np.zeros((n,m))

R_groups[0,:] = [1,1,1,1,1,1]
R_groups[1,:] = [1,1,1,1,1,1]
R_groups[2,:] = [1,1,1,1,1,1]
R_groups[3,:] = [6,1,6,1,6,1]
R_groups[4,:] = [6,1,6,1,6,1]
R_groups[5,:] = [6,1,6,1,6,1]
R_groups[6,:] = [1,6,1,6,1,6]
R_groups[7,:] = [1,6,1,6,1,6]
R_groups[8,:] = [1,6,1,6,1,6]

def demofactorise(dataset):
    p=3
   
    d=2
 
    #maxiter_main = 2
    tries = 0
    n_tries = 50
    group_convergence = 5 #age groups need to reach before convergence met
    existing_groups_prev = np.zeros(p)
    existing_groups_prev.fill(False)
    existing_groups_prev = np.ones((1,p), dtype=bool)
    groups_age = 0 
    growth_frequency = 10 #num iterations between each group growth step
    density = 0.8
    
    
    if dataset == "0":#theoretical
        R = R_groups
    else:
        R = R_groups
        
    n,m = R.shape    
    W,L,Rsampled,Rmissing,a = blc.sampleR(R, density)
    P = blc.createP(p,n) #create initial P - we don't know where users lie so this is arbitrary for now and will be learned
    tolerance = 0.01 #error tolerance to be met for factorisation...
    
    #**make update method that updates all the things each round - ie rtilde, Lambda, R, etc...
    while ( (tries < n_tries) and (groups_age < group_convergence) ):
        Rtilde, Lambda = blc.createRtilde(Rsampled,P)#avg rating per group for each item - pxm
       # Lambda = blc.createLambda(P,Rsampled)#users per group rating item m - pxm - 
        a = Lambda>0 #boolean index to only calculate items that have been rated by that group   
        Ut,Vt, memt = blc.ls_groups(Rtilde, d, tolerance, 10, Lambda, a) #factoristaion - groupwise
        #reallocate users to closest group
        for user in np.random.permutation(n):  
            blc.findP(Rsampled,Rtilde,user,P)#this updates P to put user in closest group by comparing R and Rtilde
        #for privacy, would only need that users row of R**include this!!? - this can be done locally by user - embarrassingly parallel
        if(not(tries%growth_frequency)):#update num groups and associated parameters
            P, Ut = blc.train_groups(P,Ut)#cull unused groups and then double them      
    #        #Rtilde and lambda will be updated on the beginning of the next iteration...
        tries +=1
        #group convergence
        existing_groups = np.sum(a,0) > 0 #find which groups are empty/full
        if(existing_groups_prev.size == existing_groups.size):
            if((existing_groups_prev ^ existing_groups).any()): #if they're not the same
                groups_age = 0
                existing_groups_prev = existing_groups
        else:
            groups_age += 1
            
    #delete empty groups to clean up...
    UV = np.dot(Ut.T, Vt)#factorized "guess" of Rtilde
    
    groups_used = P.sum(axis = 1)
    P = P[groups_used!=0] #get rid of empty groups
    p,n = P.shape 
    Rtilde = Rtilde[groups_used!=0]#pxm
    Lambda = Lambda[:, groups_used!=0]#mxp  
    Ut = Ut[:,groups_used!=0] 
    UV = np.dot(Ut.T, Vt)#factorized "guess" of Rtilde
    Rtilde_missing = blc.createRtilde(Rmissing, P)[0]#expected predictions to compare
    zero_elements = np.sum(Rsampled!=0)
    factorisation_error = blc.rms(Rtilde, Ut, Vt)#compare known elements used in factorization to factored results 
    prediction_error = blc.rms(Rtilde_missing, Ut, Vt)#compare missing elements (not given to recommender) to predicted values 
    
    return factorisation_error, prediction_error
    