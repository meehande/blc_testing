# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:24:26 2016

@author: Deirdre Meehan
"""
import blc_oldfromgit as blc
import numpy as np
import pandas as pd
import os
p=3
m=6
n=9
d=3

maxiter_main = 2
tries = 0
n_tries = 50
group_convergence = 3 #age groups need to reach before convergence met
existing_groups_prev = np.zeros(p)
existing_groups_prev.fill(False)
existing_groups_prev = np.ones((1,p), dtype=bool)
groups_age = 0

growth_frequency = 10 #num iterations between each group growth step
#R = blc.createR(n,m,d) #really this will be given - this is the recommendation system
"""
TEST: CREATE R WITH GROUPS TO TEST IF SAME GROUPS CONVERGE...
"""
results = pd.DataFrame(columns=['density', 'zero_elements', 'factor_error', 'predict_error','d','n','m','p'])
if os.path.isfile('Rgroups_test.pkl'):
    results = pd.read_pickle('Rgroups_test.pkl')

R_groups = np.zeros((n,m))

R_groups[0,:] = [1,1,1,5,5,5]
R_groups[1,:] = [1,1,1,5,5,5]
R_groups[2,:] = [1,1,1,5,5,5]
R_groups[3,:] = [5,5,5,1,1,1]
R_groups[4,:] = [5,5,5,1,1,1]
R_groups[5,:] = [5,5,5,1,1,1]
R_groups[6,:] = [5,5,5,1,1,1]
R_groups[7,:] = [5,5,5,1,1,1]
R_groups[8,:] = [5,5,5,1,1,1]

density = 1.0

    #for ii in xrange(50):    
for i in xrange(10):
    #density = i/10.0 + 0.1
    W,L,Rsampled,Rmissing,a = blc.sampleR(R_groups, density)
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
    #Rtilde, Lambda = blc.createRtilde(Rsampled,P)
    #PRINT CHECKS:
    print "number of iterations: ", tries
    print "density: ", density
    print "P\n", P
    #print "Rtilde\n", Rtilde
       # print "Ut\n", Ut
    #print "Vt\n", Vt
    print "Rtilde\n", Rtilde
    print "UV\n", UV
       # print "Rsampled\n", Rsampled
    #print "num zero els:  ", zero_elements
    #print "lambda\n", Lambda
    print "groups age: ", groups_age 
    print "factorisation error:  ",factorisation_error
    print "prediction error:  ",  prediction_error
    print "##############################"                                                                                                                                                                                                                                          

#results = results.append(pd.DataFrame({'density':density, 'zero_elements':zero_elements, 'factor_error':factorisation_error, 'predict_error':prediction_error,'d':d,'n':n,'m':m,'p':p},index=[0]),ignore_index=True)

#results.to_pickle('Rgroups_test.pkl')
print "Done"

"""
POSSIBLE TEST/RESULTS:
**GRAPH ERROR VS DENSITY - FACTORISATION ERROR AND PREDICTION ERROR - LESS ERROR WITH MORE DATA
**TRY DIFFERENT SIZED STARTING GROUPS, WILL IT LEARN RIGHT ONES?
**LEARN GROUPS INITIALLY RATHER THAN CREATING RANDOM P?
**COMPARE EFFICIENCY OF FINDP AND CREATEP TO KNOW WHICH IS BETTER TO USE TO BEGIN (CREATEP SHOULD BE FASTER BECAUSE NO LOOP FOR EACH USER)
"""  
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    