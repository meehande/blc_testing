# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:25:26 2016

@author: Deirdre Meehan
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
"""
SPARSE VERSION OF BLC.PY

"""

"""
a - csr
U, V - lil
Rtilde - lil
Lambda - csr
"""
def ls_groups(Rtilde,d,tolerance,maxiter, Lambda, a): 
  #(n,m) = R.shape
  p,m = Rtilde.shape
  sigma = np.finfo(float).eps #scalar
  Id = np.identity(d)
  Id = sps.eye(d)#sparse identity
   
  V = sps.rand(d, m, density = 0.5)#this gives COO sparse formation
  U = sps.rand(d, p, density = 0.5)
  V = V.tolil()#lil best for indexing and converting to csr
  U = U.tolil()  
  err = tolerance+1 #initialise to value > tolerance
  it = 0
  while((err>tolerance) & (it<maxiter)):  
    for g in xrange(p):#group by group
      if a[:,g].nnz>0: #if a is in sparse format - a is mxp
        Vg = V[:,a[:,g].data]  
        Lg = Lambda[a[:,g].data,g]  # this is m x p, if p is #groups its small-ish
        VV = Vg.dot(sps.diags(Lg.data).dot(Vg.T))#be careful of ordering of dot product and arrangement of result matrix!!
        Z = Rtilde[g,a[:,g].data].multiply(Lg.data)
        Z = sps.csr_matrix(Z)
        try:
          U[:,g] = np.expand_dims(spsl.lsqr(sigma*Id+VV, Vg.dot(Z.data))[0], axis=1) # dx1
        except:
          print('Ill conditioned matrix, fix that please in some way..')
        
    for v in xrange(m): 
      #pdb.set_trace()
      if a[v, :].nnz>0:
        Lv = sps.diags(Lambda.tolil()[v,a[v,:].data].data[0])  # this is n x n, ok if n is #groups and small-ish **either this has to be made sparse or find another way... HUGE
        Uv = U[:,a[v,:].data]  # this is d x n
        t1 = Uv.dot(Lv.dot(Uv.T)) + sigma*Id  # this is d x d
        t2 = Uv.dot(Lv.dot(Rtilde[a[v,:].data,v])) # RH multiply gives n x 1, LH d x 1
        t2 = t2.toarray()
        try:
          V[:,v]=np.expand_dims(spsl.lsqr(t1,t2)[0], axis=1)
        except:
          print('Ill conditioned matrix, fix that please in some way..')
          #pdb.set_trace()
    err = rms(Rtilde, U, V)
    it +=1

  mem = 0#U.nbytes+V.nbytes+Lambda.nbytes+Id.nbytes+Vg.nbytes+Lg.nbytes+VV.nbytes+Z.nbytes+Lv.nbytes+Uv.nbytes+t1.nbytes+t2.nbytes+a.nbytes
  return (U,V, mem) 
  
"""
GET FACTORISATION OR PREDICTION ERROR DEPENDING ON R GIVEN
-RSAMPLED FOR FACTORISATION ERROR
-RMISSING FOR PREDICTION ERROR
"""  
def rms(R,U,V): # sparse rms calculation
  totalsampled = sps.csr_matrix.sum(R!=0)#number of non zeros - num of actual ratings that exist that we need
  
  if totalsampled:
      UV = U.T.dot(V)
      existing = (R!=0) #& (UV!=0) 
      e = np.sum( np.asarray(UV[existing]-R[existing])**2 )/totalsampled
  else:
      e = 0
  return np.sqrt(e)
  
"""
CREATE GROUP RATING MATRIX RTILDE 
FROM USER RATING MATRIX R
AND GROUP ORGANISATION MATRIX P  
"""  
def createRtilde(R, P):
    p,n = P.shape
    n,m = R.shape
    Rhat = sps.lil_matrix((p,m))
    #Rhat[:] = np.nan
    #Rhat = pxm = sum of ratings for each item for all members of a group
    Rtilde = sps.lil_matrix((p,m))
    Lambda = createLambda(P,R)
    #Lambda = mxp = #users per group who have rated item m
    for item in xrange(m):#for each item
        #turn R = nxm into Rhat = pxm - sum users from same group     
        Rhat[:,item] = P.dot(R[:,item])#Rhat(:,v) = P*Rv
    #Rhat = aggregate of ratings per item for each group
        invLambda = sps.diags(Lambda[item,:].data[0])#make diagonal of #users per group who rated item
        invLambda.data = 1/invLambda.data
        Rtilde[:,item] = invLambda.dot(Rhat[:,item])
    return Rtilde, Lambda#Rtilde = pxm aggregation of R for each user
""" 
LAMBDA = MXP
NUMBER OF USERS IN GROUP P WHO RATED ITEM M
"""       
def createLambda(P,R):
    n,m = R.shape
    p,n = P.shape
    a = R!=0
    Lambda = sps.lil_matrix((m,p))
    for i in xrange(m):
        Lambda[i,:]= P[:,a[:,i].data].sum(axis=1).T
    return Lambda
    
def createP(p, n):
    P = sps.lil_matrix((p,n))
    for user in xrange(n):
        group = np.random.random_integers(0, p-1) #returns random num between 0, p-1 inclusive
        P[group, user] = 1
    return P
  
  
  
  
  
  
  
  
  
  
  