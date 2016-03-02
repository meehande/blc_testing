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
Issues:
recommendation with groups gives negative values in unknown spaces?
- I think it gives neg values for empty groups - **cull empty groups??
Note: changed "missing" values = 0, not NaN
"""

def ls(R,Rsampled,W,d,L,tolerance,maxiter, Lambda): # **create Lambda from P, **check L implementation L[g] **do not use python lists! Move to numpy for W and L!!
  # BLC
  (n,m) = R.shape
  sigma = np.finfo(float).eps
  Id = np.identity(d)
  
  Lambda=np.ones((m,n));  # this is m x n (large ?), item in row i col g is \Delta(i)_gg  **remove this and use P to compute Lambda HUGE
  
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

"""
Netflix recommneder matrix factorization - no groups/privacy
"""
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
  #Rsampled = np.empty((n,m))
  #Rmissing = np.empty((n,m))
  #Rsampled.fill(np.nan)
#  pdb.set_trace()
  #Rmissing.fill(np.nan)
  Rsampled = np.zeros((n,m))
  Rmissing = np.zeros((n,m))
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
  a = Rsampled>0      
  return (W,L,Rsampled,Rmissing, a)
       
def createLambda(P,R):
    n,m = R.shape
    p,n = P.shape
    a = R>0
    Lambda = np.zeros((m,p))
    for i in xrange(m):
        Lambda[i,:]= P[:,a[:,i]].sum(axis=1)
    return Lambda

def createP(p, n):
    P = np.zeros((p,n))
    for user in xrange(n):
        group = np.random.random_integers(0, p-1)
        P[group, user] = 1
    return P

def createRtilde(R, P):
    #W, L are like existing values in matlab implementation
    p,n = P.shape
    n,m = R.shape
    Rhat = np.empty((p,m))
    Rhat[:] = np.nan
    #Rhat = pxm = sum of ratings for each item for all members of a group
    Rtilde = np.zeros((p,m))
    Lambda = createLambda(P,R)
    #Lambda = mxp = #users per group who have rated item m
    for item in xrange(m):#for each item
        #turn R = nxm into Rhat = pxm - sum users from same group     
        Rhat[:,item] = np.dot(P, R[:,item])#Rhat(:,v) = P*Rv
    #Rhat = aggregate of ratings per item for each group
        invLambda = np.diag(Lambda[item, :])#make diagonal of #users per group who rated item
        invLambda[invLambda>0] = 1/invLambda[invLambda>0]
        Rtilde[:,item] = np.dot(invLambda,Rhat[:,item])
    return Rtilde#Rtilde = pxm aggregation of R for each user
    

def indexExistingValues(Lambda):
    m,p = Lambda.shape
    Wtilde = [] # for each group, items rated - p lists
    Ltilde = [] # for each item, group who rated it - m lists 
    #Lambda = mxp = #users in group p rating item m - find zero elements!
    for j in xrange(m):
        Ltilde.append([])
    for i in xrange(p):#for each group - column of Lambda
        Wtilde.append([])
        for j in xrange(m):#for each item - row of Lambda
            if(Lambda[j,i]!=0):#if item j has been given rating by group i
                Wtilde[i].append(j)
                Ltilde[j].append(i)
    return Wtilde, Ltilde

def ls_groups(Rtilde,d,tolerance,maxiter, Lambda, a): 
  #(n,m) = R.shape
  p,m = Rtilde.shape
  sigma = np.finfo(float).eps
  Id = np.identity(d)
   
  V = np.random.normal(size=(d, m))
  U = np.random.normal(size=(d, p))  # no point in initialising U as we're going to immediately update it. Not really: in case of ill conditioned matrix we may get an error because we won't chang some values for some iterations.....
  err = tolerance+1 #initialise to value > tolerance
  it = 0
  while((err>tolerance) & (it<maxiter)):  
    for g in xrange(p):#group by group
      if a[:,g].any():
      #if W[g]:     # check whether group is used (a row of R may be zero)              
        Vg = V[:,a[:,g]]  # this is d x m
        Lg = Lambda[a[:,g],g]  # this is m x p, if p is #groups its small-ish
        VV = np.dot(np.dot(Vg,np.diag(Lg)),Vg.T)  # this is d x d i.e. small
        Z = Rtilde[g,a[:,g]]*Lg  # this is 1 x m - element wise multiplication - Rtilde = pxm
#Rtilde = avg rating of item m by group p      
        try:
          U[:,g] = np.linalg.lstsq(sigma*Id+VV,np.dot(Vg,Z))[0] # dx1
        #U[:,g] = np.dot(np.dot(Vg,Z),np.linalg.pinv(sigma*Id+VV))
        except:
          print('Ill conditioned matrix, fix that please in some way..')
        #V = V + np.random.normal(size=(d, m))
        
    for v in xrange(m): # **add if check whether v is observed (a column of R may be zero)
      #pdb.set_trace()
      #if L[v]:
      if a[v, :].any():
        Lv = np.diag(Lambda[v,a[v, :]])  # this is n x n, ok if n is #groups and small-ish **either this has to be made sparse or find another way... HUGE
        Uv = U[:,a[v, :]]  # this is d x n
        t1 = np.dot(Uv,np.dot(Lv,Uv.T)) + sigma*Id  # this is d x d
        t2 = np.dot(Uv,np.dot(Lv,Rtilde[a[v, :],v]))  # RH multiply gives n x 1, LH d x 1
        try:
          V[:,v] = np.linalg.lstsq(t1, t2)[0]
          #V[:,v] = np.dot(t2,np.linalg.pinv(t1))
        except:
          print('Ill conditioned matrix, fix that please in some way..')
          #pdb.set_trace()
          #U = U + np.random.normal(size=(d, n))
    err -= rms(Rtilde, U, V)
    it +=1
# ** use np.sqrt(np.sum( (np.dot(U.T,V)-R)**2 )/numel(W))
#  tempmem = locals()
#  mem = sys.getsizeof(tempmem) 
  mem = U.nbytes+V.nbytes+Lambda.nbytes+Id.nbytes+Vg.nbytes+Lg.nbytes+VV.nbytes+Z.nbytes+Lv.nbytes+Uv.nbytes+t1.nbytes+t2.nbytes+a.nbytes
  #print(U.nbytes,V.nbytes,Lambda.nbytes,Id.nbytes,Vg.nbytes,Lg.nbytes,VV.nbytes,Z.nbytes,Lv.nbytes,Uv.nbytes,t1.nbytes,t2.nbytes,sys.getsizeof(L),sys.getsizeof([W]))
  return (U,V, mem) 
  
"""
euclidean distance between matrices A, B
dist = sqrt ( ||A||^2 + ||B||^2 - 2*A.B )
"""  
def distance(a,b):
    aa = np.sum(np.square(a),0)
    bb = np.sum(np.square(b),0)
    ab = np.dot(a.T,b)
    return np.sqrt(abs(np.column_stack([aa]*bb.shape[0])+np.vstack([bb]*aa.shape[0])-2*ab))
    
  

def train_groups(P, Ut): # expand tree in matlab code!
#1.find unused groups and get rid of them - done
#2.double number of existing groups - 
#3. update corresponding variables - eg R, Lambda...
    distgroups = 0.2
    
    groups_used = P.sum(axis = 1)
    if groups_used.any():
        P = P[groups_used!=0] #get rid of empty groups
        Ut = Ut[:,groups_used!=0]
        
        #**to do: finish this - tempm and tempk...
        #validate it works... etc
    p,n = P.shape # culled P shape needed
    tempk =  np.triu(distance(Ut,Ut)*(1-np.eye(p)))
    tempk[tempk==0] = np.inf
    tempm = tempk.min()
    if (tempm == np.inf):
        tempm = 0.1 #degenerate case - only one group - distance would be infinite so fix std deviation val
        #set values in new groups
    Ut = np.hstack((Ut, np.zeros((Ut.shape))))#double Ut - num groups
    P = np.vstack((P, np.zeros((P.shape))))#double num groups **should we just use createP??
    newd, newp = Ut.shape #shape of ut with groups culled and then doubled
    for i in range(p,newp):
        Ut[:,i] = np.random.multivariate_normal(Ut[:,i], (tempm*distgroups)**2/newd*np.eye(newd))
    return P,Ut

def rFromRtilde(Rt, P, R):
    #take avg rating of that group for that item?
    """
    sth like:
    for unrated item m by person n:
    R[n,m] = Rtilde[P[:,n]!=0, m]
    - select group they are in, give back that avg
    how to do this without going elementwise??
    """
   
 
"""
update user's group by seeing which group's ratings
are closest
"""
def findP(R, Rtilde, user, P):
    a = R>0
    distance = np.sum(np.square(R[user, a[user, :]] - Rtilde[:,a[user,:]]),1) # column of distance from user to each group
    perm = np.random.permutation(distance.size) # random permutation of indices in distance
    index = np.argmin(distance[perm])
    index = perm[index]
    P[:,user] = np.zeros(P.shape[0])
    P[index, user] = 1 #P is passed by reference
    #return P
    

def rms(Rmissing,U,V): # metric on the missing one
  # root mean square prediction error
  #print np.dot(U.T,V)-R
  totalsampled = np.sum(Rmissing!=0)
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
     (W,L,Rsampled,Rmissed, a) = sampleR(R,0.3)  # sample from it **rho
     P = createP(p,n)
     Lambda = createLambda(P,Rsampled)
     Rtilde = createRtilde(Rsampled,P)
     a = Lambda>0
     (U,V,mem) = ls(R,Rsampled,W,d,L,0.0000001,10, Lambda) # factorize
     #Wt, Lt = indexExistingValues(Lambda)
     Ut, Vt, memt = ls_groups(Rtilde, d, 0.0000001,10, Lambda,a)
     print "R\n", R
     print "Rsampled\n", Rsampled
     print "P\n", P
     print "Lambda\n", Lambda
     print "Rtilde\n", Rtilde
     print "Utilde\n", Ut
     print "Vtilde\n", Vt
     print "ls_groups\n", np.dot(Ut.T, Vt)
     print "ls\n", np.dot(U.T, V)
     #factorization error - compare elements of R_sampled - R_tilde
     #prediction error - compare elements of R_missing - R_tilde'
     e = rms(R,U,V)
     e_groups = rms(Rtilde,Ut,Vt)
     print "error is: ", e
     print "group error is: ", e_groups     
    # self.assertTrue(e<1e-3,'Accuracy is '+ str(e)) # is solution accurate ? ** change this: it shouldn't fail if accuracy is low (or remove it)

  def test_accuracy(self):
     for i in xrange(4): # try for 10 different random R matrices
        self.accuracy(20,10,3,3)

############################
if __name__ == '__main__':
   unittest.main(verbosity=2)
# to run from python:
#suite =  unittest.TestLoader().loadTestsFromTestCase(TestBLC)
# unittest.TextTestRunner(verbosity=3).run(suite)



