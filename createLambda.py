# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:41:15 2016

@author: Deirdre Meehan
"""
import numpy as np
import unittest

def createLambda(P, R):
    n,m = R.shape
    p,n = P.shape
    Lambda = np.zeros((m,p))
    
    for item in xrange(m):
        Pu = np.empty(p)#initialise this to fit next calc
        for user in xrange(n):
            if((np.isnan(R[user, item])) == False):#if user has rated this item
                Pu = np.c_[Pu,P[:,user]]#append the column from P for that user
        Pu = np.delete(Pu,0,1)#get rid of initial col of P - used to initialise variable
        delta_v = np.dot(Pu, Pu.T) #compute delta as P.P^T for each item - pxp
        lam_v = []
        for row in xrange(p): 
            lam_v.append(sum(delta_v[row,:]))#1xp
        Lambda[item,:]=lam_v
        #convert list of lists to matrix
    #Lambda = np.array([np.array(li) for li in Lambda])
    return Lambda


def createP(p, n):
    P = np.zeros((p,n))
    for user in xrange(n):
        group = np.random.random_integers(0, p-1)
        P[group, user] = 1
    return P
"""
Issues:
P is not always giving one group per user
Test is generating n = 20? - even though lambda is getting the right dimension
Pu didn't delete first column
delta_v is not coming out as a diagonal matrix - should it definitely be? I think so
"""    


##################
"""
TO DO:
**Finish Test
**Put into ls
**Test in blc
**createP 
**run some performance tests
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
       
                