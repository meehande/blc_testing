import numpy as np
import unittest
import pdb
import sys
#from memory_profiler import profile #add decorator @profile before function to check and then just run the file

# compatibility python 3
try:
    xrange
except NameError:
    xrange = range

'''This is for 
def numel(lis):
  sum(len(x) for x in lis)
'''

"""
To Do:
**put lambda from p in ls
-make lambda fit dimension wise - r in range(p) now?
-lambda = mxp
-as a diagonal it is pxp (used when multiplying)
-does r, u, v have p in their dimension? (ie rn p = n but change to p everywhere?)
"""

def ls(R,Rsampled,W,d,L,tolerance,maxiter, Lambda): # **create Lambda from P, **check L implementation L[g] **do not use python lists! Move to numpy for W and L!!
  # BLC
  (n,m) = R.shape
  sigma = np.finfo(float).eps
  Id = np.identity(d)
  
 # Lambda=np.ones((m,n));  # this is m x n (large ?), item in row i col g is \Delta(i)_gg  **remove this and use P to compute Lambda HUGE
  
  V = np.random.normal(size=(d, m))
  U = np.random.normal(size=(d, n))  # no point in initialising U as we're going to immediately update it. Not really: in case of ill conditioned matrix we may get an error because we won't chang some values for some iterations.....
  #U = np.zeros(shape=(d,n))
  err = tolerance+1 #initialise to value > tolerance
  it = 0
  #for i in xrange(10): #**change this with a while loop and check whether the factorisation (no prediction) error is less than tolernace to stop.
  while((err>tolerance) & (it<maxiter)):  
    for g in xrange(n): # **add if: check whether group is used (a row of R may be zero)
      if W[g]:      
        Vg = V[:,W[g]]  # this is d x m
        Lg = Lambda[W[g],g]  # this is m x n, if n is #groups its small-ish
        VV = np.dot(np.dot(Vg,np.diag(Lg)),Vg.T)  # this is d x d i.e. small
        Z = R[g,W[g]]*Lg  # this is 1 x m - element wise multiplication - this is supposed to represent Rtilde?? - it doesn;t though...
#if Z = rtilde **need to make Lg divide R[g,:] by #users per group - element wise inversion        
        try:
          U[:,g] = np.linalg.lstsq(sigma*Id+VV,np.dot(Vg,Z))[0] # dx1
        #U[:,g] = np.dot(np.dot(Vg,Z),np.linalg.pinv(sigma*Id+VV))
        except:
          print('Ill conditioned matrix, fix that please in some way..')
        #V = V + np.random.normal(size=(d, m))
        
    for v in xrange(m): # **add if check whether v is observed (a column of R may be zero)
      #pdb.set_trace()
      if L[v]:
        Lv = np.diag(Lambda[v,L[v]])  # this is n x n, ok if n is #groups and small-ish **either this has to be made sparse or find another way... HUGE
        Uv = U[:,L[v]]  # this is d x n
        t1 = np.dot(Uv,np.dot(Lv,Uv.T)) + sigma*Id  # this is d x d
        t2 = np.dot(Uv,np.dot(Lv,R[L[v],v]))  # RH multiply gives n x 1, LH d x 1
        try:
          V[:,v] = np.linalg.lstsq(t1, t2)[0]
          #V[:,v] = np.dot(t2,np.linalg.pinv(t1))
        except:
          print('Ill conditioned matrix, fix that please in some way..')
          #pdb.set_trace()
          #U = U + np.random.normal(size=(d, n))
    err = rms(Rsampled, U, V)
    it +=1
# ** use np.sqrt(np.sum( (np.dot(U.T,V)-R)**2 )/numel(W))
#  tempmem = locals()
#  mem = sys.getsizeof(tempmem) 
  mem = U.nbytes+V.nbytes+Lambda.nbytes+Id.nbytes+Vg.nbytes+Lg.nbytes+VV.nbytes+Z.nbytes+Lv.nbytes+Uv.nbytes+t1.nbytes+t2.nbytes+sys.getsizeof(L)+sys.getsizeof([W])
  #print(U.nbytes,V.nbytes,Lambda.nbytes,Id.nbytes,Vg.nbytes,Lg.nbytes,VV.nbytes,Z.nbytes,Lv.nbytes,Uv.nbytes,t1.nbytes,t2.nbytes,sys.getsizeof(L),sys.getsizeof([W]))
  return (U,V,mem)

def ls2(R,W,d,L):
  # netflix baseline
  (n,m) = R.shape
  sigma = 0.0001
  V = np.random.rand(d,m)
  U = np.zeros((d,n))
  Id = np.identity(d)
  Lambda=np.ones((m,n));  # item in row i col g is \Delta(i)_gg
  
  for i in xrange(100):
    for g in xrange(n):
        Vg = V[:,W[g]]
        
        # min (V'U - R')^2 by solving inv(VV')VR'
        t1 = np.dot(Vg, Vg.T) + Id*sigma
        t2 = np.dot(V, R[g,W[g]].T)
        U[:,g] = np.linalg.solve(t1,t2)

    for v in xrange(m):
        Ug = U[:,L[v]]
        # min (U'V - R)^2 by solving inv(UU')UR
        t1 = np.dot(Ug,Ug.T) + sigma*Id
        t2 = np.dot(Ug,R[L[v],v])
        V[:,v] = np.linalg.solve(t1, t2)
  return (U,V)


def createR(n,m,d):
  # generate random user-item rating matrix
  Ustar = np.random.rand(d, n)
  Vstar = np.random.rand(d, m)
  return np.dot(Ustar.T,Vstar)


def sampleR(R,density):
  # sample from R. just now we take all elements
  (n,m) = R.shape
  sampled = np.random.uniform(size=(n,m)) < density #it's a boolean matrix that tells us whether the value is sampled or not
  Rsampled = np.empty((n,m))
  Rmissing = np.empty((n,m))
  Rsampled.fill(np.nan)
#  pdb.set_trace()
  Rmissing.fill(np.nan)
  W=[]
  L=[]
  missingW = []
  missingL = []
  for j in xrange(m):#for each column in R add a list to L
    L.append([]) 
    missingL.append([])
  for i in xrange(n):#for each row in R add a list to W
    W.append([])
    missingW.append([])
    for j in xrange(m):
      if sampled[i,j] and np.isfinite(R[i,j]): # check if value exists (assuming that we use NaNs or Inf to represent missing values, change accordingly otherwise!
        W[i].append(j)
        L[j].append(i)
        Rsampled[i,j] = R[i,j]
      else:
        missingW[i].append(j)
        missingL[j].append(i)
        Rmissing[i,j] = R[i,j]
 # totalsampled = np.sum(np.isfinite(Rsampled))
  #pdb.set_trace()
  return (W,L,Rsampled,Rmissing)

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
    return Lambda #mxp
    
def createP(p, n):
    P = np.zeros((p,n))
    for user in xrange(n):
        group = np.random.random_integers(0, p-1)
        P[group, user] = 1
    return P

def rms(Rmissing,U,V): # metric on the missing one
  # root mean square prediction error
  #print np.dot(U.T,V)-R
  totalsampled = np.sum(np.isfinite(Rmissing))
  if totalsampled:
      UV = np.dot(U.T,V)
      e = np.nansum( (UV-Rmissing)**2 )/totalsampled
  else:
      e = 0
  return np.sqrt(e)

############################
class TestBLC(unittest.TestCase):

  def accuracy(self,n,m,d,p):
     R = createR(n,m,d)  # generate random user-item rating matrix
     (W,L,Rsampled,Rmissed) = sampleR(R,1)  # sample from it **rho
     P = createP(n,n)
     Lambda = createLambda(P,Rsampled)
     (U,V,mem) = ls(R,Rsampled,W,d,L,0.0000001,10, Lambda) # factorize
     e = rms(Rmissed,U,V)
     self.assertTrue(e<1e-3,'Accuracy is '+ str(e)) # is solution accurate ? ** change this: it shouldn't fail if accuracy is low (or remove it)

  def test_accuracy(self):
     for i in xrange(4): # try for 10 different random R matrices
        self.accuracy(20,4,3,2)

############################
if __name__ == '__main__':
   unittest.main(verbosity=2)
# to run from python:
#suite =  unittest.TestLoader().loadTestsFromTestCase(TestBLC)
# unittest.TextTestRunner(verbosity=3).run(suite)



