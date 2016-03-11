import numpy as np
import unittest
import sys

def ls(R,W,d,L): # **create Lambda from P, **check L implementation L[g]
  # BLC
  (n,m) = R.shape
  sigma = 0.0001
  Id = np.identity(d)
  
  Lambda=np.ones((m,n));  # this is m x n (large ?), item in row i col g is \Delta(i)_gg  **remove this and use P to compute Lambda
  
  V = np.random.rand(d, m)
  U = np.empty((d,n))  # no point in initialising U as we're going to immediately update it 
  for i in xrange(100):
    for g in xrange(n): # **add if: check whether group is used (a row of R may be zero)
        Vg = V[:,W[g]]  # this is d x m
        Lg = Lambda[W[g],g]  # this is m x n, if n is #groups its small-ish
        VV = np.dot(np.dot(Vg,np.diag(Lg)),Vg.T)  # this is d x d i.e. small
        Z = R[g,W[g]]*Lg  # this is 1 x m
        U[:,g] = np.linalg.solve(sigma*Id+VV,np.dot(Vg,Z))
   
    for v in xrange(m): # **add if check whether v is observed (a column of R may be zero)
        Lv = np.diag(Lambda[v,L[v]])  # this is n x n, ok if n is #groups and small-ish
        Uv = U[:,L[v]]  # this is d x n
        t1 = np.dot(Uv,np.dot(Lv,Uv.T)) + sigma*Id  # this is d x d
        t2 = np.dot(Uv,np.dot(Lv,R[L[v],v]))  # RH multiply gives n x 1, LH d x 1
        V[:,v] = np.linalg.solve(t1, t2)
  tempmem = locals()
  mem = sys.getsizeof(tempmem) 
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
    for g in range(n):
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
  W=[]
  L=[]
  for j in range(m):
    L.append([]) 
  for i in range(n):
    W.append([])
    for j in range(m):
       if sampled[i,j] and np.isfinite(R[i,j]): # check if value exists (assuming that we use NaNs or Inf to represent missing values, change accordingly otherwise!
          W[i].append(j)
          L[j].append(i)
  return (W,L)

def rms(R,U,V): # do we need this metric or only predicted ones?
  # root mean square prediction error
  #print np.dot(U.T,V)-R
  e = np.mean( (np.dot(U.T,V)-R)**2 )
  return np.sqrt(e)

############################
class TestBLC(unittest.TestCase):

  def accuracy(self,n,m,d):
     R = createR(n,m,d)  # generate random user-item rating matrix
     (W,L) = sampleR(R,1)  # sample from it
     (U,V,mem) = ls(R,W,d,L) # factorize
     e = rms(R,U,V)
     self.assertTrue(e<1e-3,'Accuracy is '+ str(e)) # is solution accurate ? ** change this: it shouldn't fail if accuracy is low (or remove it)

  def test_accuracy(self):
     for i in range(10): # try for 10 different random R matrices
        self.accuracy(100,10,5)

############################
if __name__ == '__main__':
   unittest.main(verbosity=2)




