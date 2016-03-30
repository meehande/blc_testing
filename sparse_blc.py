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
          U[:,g] = np.expand_dims(spsl.lsmr(sigma*Id+VV, Vg.dot(Z.data))[0], axis=1) # dx1
        except:
          print('Ill conditioned matrix, fix that please in some way..')
        
    for v in xrange(m): 
      #pdb.set_trace()
      if a[v, :].nnz>0:
        Lv = sps.diags(np.asarray(Lambda[v,a[v,:].data].todense())[0])  # this is n x n, ok if n is #groups and small-ish **either this has to be made sparse or find another way... HUGE
        Uv = U[:,a[v,:].data]  # this is d x n
        t1 = Uv.dot(Lv.dot(Uv.T)) + sigma*Id  # this is d x d
        t2 = Uv.dot(Lv.dot(Rtilde[a[v,:].data,v])) # RH multiply gives n x 1, LH d x 1
        t2 = t2.toarray()
        try:
          V[:,v]=np.expand_dims(spsl.lsmr(t1,t2)[0], axis=1)
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
        invLambda = Lambda[item,:].data[0] #list of that row of lambda
        while len(invLambda) < p:
            invLambda.append(0)
        invLambda = sps.diags(invLambda)#make diagonal of #users per group who rated item
        invLambda.data = 1/invLambda.data
        Rtilde[:,item] = invLambda.dot(Rhat[:,item])
    Rtilde[Rtilde==np.inf] = 0
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
    
"""
P = GROUP STRUCTURE = NXP
PUT EVERY USER IN A RANDOM GROUP - DIAGONAL MATRIX?
"""    
def createP(p, n):
    P = sps.lil_matrix((p,n))
    for user in xrange(n):
        group = np.random.random_integers(0, p-1) #returns random num between 0, p-1 inclusive
        P[group, user] = 1
    return P
  
  
"""
euclidean distance between matrices A, B
dist = sqrt ( ||A||^2 + ||B||^2 - 2*A.B )
- from matlab implemetation by Roland Bunschoten in original matlab code
"""  
def distance(a,b):
    aa = a.power(2).sum(0)#square all elements, sum each column
    bb = b.power(2).sum(0)
    ab = a.T.dot(b).todense()
    #the above all return matrices-same as non sparse implementation so don't use sparse from here!
    return np.sqrt(abs(np.column_stack([aa]*bb.shape[0])+np.vstack([bb]*aa.shape[0])-2*ab))
    #below didn't return consistent results with non-sparse eqv
    #sps.csr_matrix(abs(sps.vstack([aa]*bb.shape[0]).T + sps.vstack([bb]*aa.shape[0]) - 2*ab)).sqrt()
    
    
"""
PRUNE GROUPS:
1.find unused groups and get rid of them 
2.double number of existing groups 
3.return updated P and Ut
"""
def train_groups(P, Ut): #expand tree in matlab code!
    distgroups = 0.2
    groups_used = P.sum(axis = 1).T#sum each row - matrix result with each row = #members of group - 1xp
    groups_used = np.asarray(groups_used)[0]
    if groups_used.any():
        P = P[groups_used!=0] #get rid of empty groups
        Ut = Ut[:,groups_used!=0]
    
    p,n = P.shape # culled P shape needed
    tempk =  np.triu(distance(Ut,Ut)*(1-np.eye(p)))#this is not sparse!! (distance returns dense MATRIX)
    tempk[tempk==0] = np.inf #tempk = pxp - p is likely small enough for non sparse (<<<100)
    tempm = tempk.min()
    if (tempm == np.inf):
        tempm = 0.1 #degenerate case - only one group - distance would be infinite so fix std deviation val
    #set values in new groups
    Ut = sps.hstack((Ut,  sps.lil_matrix(Ut.shape))).tolil()#double Ut - num groups
    P = sps.vstack((P, sps.lil_matrix(P.shape))).tolil()#double num groups **should we just use createP??
    newd, newp = Ut.shape #shape of ut with groups culled and then doubled
    for i in range(p,newp):
        Ut[:,i] = sps.lil_matrix(np.random.multivariate_normal(np.asarray(Ut.todense())[:,i], (tempm*distgroups)**2/newd*np.eye(newd))).T
    return P,Ut
    
    
"""
LEARN GROUPS
update (one) user's group by seeing which group's ratings
are closest
"""
def findP(R, Rtilde, user, P): #**make it work for empty param P given as input
    a = R!=0#csr format
    distance = (sps.vstack(([R[user,a[user,:].data]]*Rtilde.shape[0] )) - Rtilde[:,a[user,:].data]).power(2).sum(1)# column of distance from user to each group
    #distance=matrix                      #1xm - pxm    
    perm = np.random.permutation(distance.size) # random permutation of indices in distance )array_
    index = np.argmin(distance[perm])
    index = perm[index]
    P[:,user] = 0
    P[index, user] = 1 #P is passed by reference
    #return P

    
   
  
  
  
  
  
  
  
  
  