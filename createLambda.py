# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:41:15 2016

@author: Deirdre Meehan
"""
import numpy as np
import unittest

def createLambda2(P, R):
    n,m = R.shape
    p,n = P.shape
    Lambda = np.zeros((m,p))
    
    for item in xrange(m):
        Pu = np.zeros(p)#initialise this to fit next calc
        #need to test that any ratings exist in row first -- because bug found where it broke for unrated item!!
        if(sum(R[:,item]) != 0): """ are unrated items zero or NaN?! """ 
            for user in xrange(n):#**try get rid of nested loops!
                if(R[user, item] != 0):#if user has rated this item
                    Pu = np.c_[Pu,P[:,user]]#append the column from P for that user
            Pu = np.delete(Pu,0,1)#get rid of initial col of P - used to initialise variable
            delta_v = np.dot(Pu, Pu.T) #compute delta as P.P^T for each item - pxp
            lam_v = []
            for row in xrange(p): 
                lam_v.append(sum(delta_v[row,:]))#1xp
            Lambda[item,:]=lam_v
        else: #if item is unrated by anyone!
            Lambda[item, :] = np.zeros(p)
        #convert list of lists to matrix
    #Lambda = np.array([np.array(li) for li in Lambda])
    return Lambda


def createP(p, n):
    P = np.zeros((p,n))
    for user in xrange(n):
        group = np.random.random_integers(0, p-1)
        P[group, user] = 1
    return P
    
def updateP(R, P, W, L):
    #W, L are like existing values in matlab implementation
    p,n = P.shape
    n,m = R.shape
    Rhat = np.empty((p,m))
    Rhat[:] = np.nan
    #Rhat = pxm = sum of ratings for each item for all members of a group
    Rtilde = np.zeros((p,m))
    Lambda = createLambda2(P,R)
    #Lambda = mxp = #users per group who have rated item m
    for item in xrange(m):#for each item
        #turn R = nxm into Rhat = pxm - sum users from same group     
        Rhat[:,item] = np.dot(P, R[:,item])#Rhat(:,v) = P*Rv
    #Rhat = aggregate of ratings per item for each group
        invLambda = np.diag(Lambda[item, :])#make diagonal of #users per group who rated item
        invLambda[invLambda>0] = 1/invLambda[invLambda>0]
        Rtilde[:,item] = np.dot(invLambda,Rhat[:,item])
    return Rtilde#Rtilde = pxm aggregation of R for each user
    
    
"""
 TO DO:
 relate existingvalues to W/L
 look at notes made to finish making rtilde
 **make blc work 
"""       
    
    
"""
 ISSUES:
 Lambda should use L/W instead of R... makes sense (see matlab code)?
"""
"""
This is not giving the same result as the above function 
- not consistently #of users per group all the time and I don't know why yet
"""
def createLambda(P,L,W):
    p,n = P.shape
    m = len(L)
    #L = m lists (column-wise through R) - for each item, who has rated it
    #W = n lists (row-wise through R) - for each user, what items have they rated
    Lambda = np.zeros((m,p))
    
    for item in xrange(m):
        Pu = np.empty(p)
        #print "item ", item
        for user in xrange(n):
           # print "user ", user
            if(item in W[user]):#has this user rated this item
                Pu = np.c_[Pu, P[:,user]]
        Pu = np.delete(Pu,0,1)
        delta_v = np.dot(Pu, Pu.T) #compute delta as P.P^T for each item - pxp
        lam_v = []
        for row in xrange(p): 
            lam_v.append(sum(delta_v[row,:]))#1xp
        Lambda[item,:]=lam_v
    #convert list of lists to matrix
    #Lambda = np.array([np.array(li) for li in Lambda])
    return Lambda
        
            
             
     


##################
"""
TO DO:
**run some performance tests
**fix test to make it more generic so it doesn't fail when changed
"""

class TestFunction(unittest.TestCase):
    def test_createLambda(self):
        P = createP(3,4)
        print "P\n", P
        R = np.array([[np.nan, 0.76],[0.65, np.nan],[0.99, 0.87], [0.34, np.nan]])
        print "R\n", R        
        Lam = createLambda(P,R)
        print "Lam\n",Lam
       # Lam_correct = np.array([[1.0,1.0,1.0],[1.0,0.0,1.0]])
      #  self.assertEqual(Lam,Lam_correct)
        
############################
if __name__ == '__main__':
   unittest.main(verbosity=2)
       
                