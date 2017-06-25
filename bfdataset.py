import math, random
from math import sqrt,ceil
import numpy as np

## dataset  class
#* generates test data sets with different properties:
#    * 2D Grid of grids: grid(grid1=True, grid2=True)
#    * 2D Grid of Gaussians: grid(grid1=True, grid2=False, sig=...)
#    * 1D Grid of grids: grid1D(grid1=True, grid2=True)
#    * 1D Grid of Gaussians: grid1D(grid1=True, grid2=False, sig=...)
#    * 1D or 2D Mixture of Gaussians: gaussmix()
#* provides initalization of a codebook with different methods
#    * randomly from the data set without repetition: randomInit()
#    * k-means++ initialization: kmeansplusplusInit()

# compute squared error of given codebook and given dataset
def errorOf(C,X):
    mat = np.zeros([X.shape[0],C.shape[0]])
    for i in range(len(C)):
        #di = (X-C[i])
        mat[:,i] = (np.linalg.norm(X-C[i],axis=1))
    return sum(np.min(mat,axis=1)**2)

class MyProblem(Exception):
    pass

class dataset:
    def __init__(self, n=1000,d=2,g=10,sig=0.1,grid1=False,grid2=False,relsize=0.5):
        self.n=n # number of data points
        self.d=d # dimensionality of data
        self.d2=2 # returned dimensionality of data (always 2)
        self.g=g # number of clusters in data set
        self.sig=sig
        self.grid1=grid1 # wether to have macro grid (clusters arranged in a grid shape)
        self.grid2=grid2 # whether to have micro grid (cluster *have* grid shape)
        self.relsize=relsize
        self.isSquare = False
        self.data=None
        self.create() # create data
        self.ratio=0 # if > 0 can be used to derive k from g
        pass
    
    @classmethod
    # 2D grid data set where each cluster can be be a grid as well
    def grid(cls, so, si, ratio=None,grid1=True,grid2=True,sig=0.1,relsize=0.5): # create dataset with a so*so grid of si*si clusters
        g = so**2
        n = g*si**2
        if so==1:
            obj = cls(n=n,d=2,g=g,grid1=grid1,grid2=grid2,sig=sig,relsize=1.0)
        else:
            obj = cls(n=n,d=2,g=g,grid1=grid1,grid2=grid2,sig=sig,relsize=relsize)            
        if ratio:
            obj.ratio=ratio
            obj.k=g*ratio
        obj.create() # create data
        return obj
    
    # create 1D dataset with a g clusters of si points (n=g*si)
    @classmethod
    def grid1D(cls, g, si, ratio=1,grid1=True,grid2=True,sig=0.1,relsize=0.5):
        n = g*si
        obj = cls(n=n,d=1,g=g,grid1=grid1,grid2=grid2,sig=sig, relsize=relsize)
        obj.k=g*ratio
        obj.ratio=ratio
        obj.create() # create data
        return obj
    
    @classmethod
    # create 2D dataset of n points distributed to g Gaussian clusters of std dev sigma
    def gaussmix(cls, n,g,sig, k=None,grid1=False,grid2=False):        
        obj = cls(n=n,d=2,g=g,grid1=grid1,grid2=grid2,sig=sig)
        if k:
            obj.k=k
        obj.create() # create data
        return obj
    
    @classmethod
    def gaussmixXD(cls,n,d,g,sig):
        #print(n,d,g,sig)
        obj = cls(n=n,d=d,g=g,sig=sig)
        obj.create() # create data
        return obj
        
    def set_k(self,k,ratio=10,ensureSquare=False):
        # set k and n based on ratio .....
        self.k = k
        self.n = k*ratio
        
    def set_ratio(self,ratio):
        self.ratio = ratio # how many centers per cluster in the optimal solution
        self.g = int(k/ratio)
        
    def ensureSquare(self):
        # ensure that the number of points in a cluster is square
        # increase n if necessary to acchieve that
        if not self.grid2:
            return
        ppc= int(math.ceil(self.n/self.g))
        self.n=ppc*self.g
        loo=0
        while(True and loo<10):
            loo+=1
            x=int(math.floor(math.sqrt(ppc)))
            if x*x==ppc and ppc*self.g==self.n:
                self.n=ppc*self.g
                print("n is now:",self.n)
                break
            else:
                print("ppc=",ppc,"n=", self.n, "g=",self.g)
                ppc+=1
                self.n=ppc*self.g
        self.isSquare = True
        
    def randomInit(self,k=None): # get codebook of size k at random from dataset
        if k is not None:
            self.k = k
        if self.data.any():
            a = list(range(self.n))
            random.shuffle(a)
            self.ibook=np.zeros((self.k, self.d2)) # ibook definition
            for i in range(self.k):
                self.ibook[i]=self.data[a[i]]
            self.ierror=errorOf(self.ibook,self.data)
            return self.ibook
        else:
            raise "no data hhhh"
            
    def kmeansplusplusInit(self, k = None): # adapted from scipy implementation
        if self.data.any():
            if k is not None:
                if k > self.data.shape[0]:
                    raise MyProblem("k too large:"+str(k)+" for datasize:"+str(self.data.shape[0]))
                else:
                    self.k = k
            elif self.k is not None:
                k=self.k
            else:
                raise MyProblem("kmeansplusplusInit no k whatsoever defined")
            n_local_trials = 2 + int(np.log(self.k))# taken from scikit
            #print("n_local_trials=",n_local_trials)
            self.ibook=np.zeros((self.k, self.d2))
            center_id = random.randint(0,self.n-1) # choose initial center
            self.ibook[0]=self.data[center_id] #store in in codebook
            
            # substract current center from all data points
            delta = self.data-self.ibook[0]
            # compute L2 norm
            best_dis = np.linalg.norm(delta,axis=1) #closest so far for all data points
            best_dis = best_dis**2 # square distance
            best_pot = best_dis.sum() # best SSE
            for c in range(1,self.k):
                rand_vals=np.random.random(n_local_trials)*best_pot
                candidate_ids = np.searchsorted(best_dis.cumsum(), rand_vals)
                # Decide which candidate is the best
                best_candidate = None
                best_pot = None
                for trial in range(n_local_trials):
                    # substract current candidate from all data points
                    delta_curcand=self.data-self.data[candidate_ids[trial]]
                    # compute L2 norm
                    dis_curcand=np.linalg.norm(delta_curcand,axis=1)
                    dis_curcand=dis_curcand**2
                    # take minimum (must be smaller or equal than previous)
                    dis_cur=np.minimum(best_dis,dis_curcand)
                    #print("minimum:",dis_cur)
                    new_pot=dis_cur.sum() # resulting potential
                    # Store result if it is the best local trial so far
                    if (best_candidate is None) or (new_pot < best_pot):
                        best_candidate = candidate_ids[trial]
                        best_pot = new_pot
                        cand_dis = dis_cur
                best_dis=cand_dis
                self.ibook[c]=self.data[best_candidate] # ibook contains vectors in order of placement!
            self.ierror=errorOf(self.ibook,self.data)
            return self.ibook
        else:
            raise "no data present"
            
    def create(self):
        if self.d > 2:
            # only GMM implemented for d>2 (i.e. no grid at this point)
            self.data, self.centers  = getGMMData(n=self.n, d=self.d, g=self.g,sig=self.sig)
        elif self.d == 2:
            # 2D data
            self.data, self.centers, self.scale = getRandomData2D(n=self.n, d=self.d, g=self.g,sig=self.sig,
                                                      grid1=self.grid1,grid2=self.grid2,relsize=self.relsize)
        elif self.d == 1:
            # 1D data
            self.data, self.centers, self.scale = getRandomData1D(self.n,self.g,self.sig,grid1=True,grid2=True,relsize=self.relsize)
        else:
            raise ValueError
            
    def getData(self):
        return self.data

def getGMMData(n,d,g,sig):
    #
    # returns n data points from a mixture of k Gaussians with covariance sig*unity
    # Gaussian centers are from uniform random in the k-d unit square
    #
    mean=np.zeros(d)
    cov=np.identity(d)*sig # symmetric
    #print("cov=", cov)
    centers = np.random.random([g,d])
    sizes=np.array([int(n/g)]*g)
    # fix last one
    sizes[-1]=n-sum(sizes[:g-1])
    #print("sizes=", sizes)
    X = None
    # make some clusters in the data set
    for i in range(g): 
        cdata = centers[i]+np.random.multivariate_normal(mean, cov, sizes[i])
        if X is not None:
            X = np.concatenate((X,cdata))
        else:
            X = cdata
    return X, centers

# 1. Test Data Generator (from Mixture of Gaussians or rectangular grid)
## 2-D grid
# arrange the point in X on a square grid (as square as possible for the given number)
def griddify(x,scale=1.0,center=False):
    g = x.shape[0] # number of points
    a = int(ceil(sqrt(g))) # x side length of grid
    b = int(ceil(g/a)) # y side length of grid
    z=0
    for i in range(a):
        for j in range(b):
            if z >= g:
                # max number of points reached (can happen if g<a*b)
                break
            else:
                x[z][0]=(i+0.5)*(1.0/(a)) # centered in unit square
                x[z][1]=(j+0.5)*(1.0/(b)) # centered in unit square
                z+=1
        else:
            continue
        break
    if center:
        for i in range(g):
            x[i][0]-=0.5
            x[i][1]-=0.5
    x = x*scale # overall scale if desired  
    return x,a,b
    
    
def getRandomData2D(n,d,g,sig,grid1=False, grid2=False,relsize=0.5):
    #
    # returns n data points from 
    # a) a mixture of g Gaussians with covariance sig*unity (grid2=False)
    #   or
    # b) a mixture of g local clusters of data points with a square-like grid shape (grid2 = True)
    #
    # g cluster centers are taken from either
    # a) a uniform random distribution in the k-d unit square (grid=False)
    #   or
    # b) a square like grid with maximally square dimensions (really sqare if g is a square number)
    #print("getRandomData2D(n=",n,",d=",d,",g=",g,",sig=",sig,",grid1=",grid1,",grid2=",grid2,",relsize=",relsize,")")
    mean=np.zeros(d)
    cov=np.identity(d)*sig # symmetric
    if grid1:
        # cluster centers in a grid
        centers = np.zeros([g,d])
        centers,a,b = griddify(centers)
    else:
        # cluster centers at uniform random positions
        centers = np.random.random([g,d])
        a = int(ceil(sqrt(g))) # 
        
                
    sizes=np.array([int(n/g)]*g)
    # fix last one
    sizes[-1]=n-sum(sizes[:g-1])
    #print("sizes=", sizes)
    X = None
    # make some clusters in the data set
    scale=0
    for i in range(g): 
        if grid2:
            newdat = np.zeros((sizes[i],d))
            scale = 1.0/(a)*relsize
            #print("scale = ", scale)
            newdat,_,_ = griddify(newdat,scale=scale,center=True)
        else:
            newdat = np.random.multivariate_normal(mean, cov, sizes[i])

        cdata = centers[i]+ newdat
        if not X is None:
            X = np.concatenate((X,cdata))
        else:
            X = cdata
    return X, centers, scale


# 1-D grid
# arrange the point in X on a square grid (as square as possible for the given number)
def griddify1D(x,scale=1.0,center=False):
    g = x.shape[0]
    for i in range(g):
        x[i][0]=(i+1)*(1.0/(g+1))
    if center:
        for i in range(g):
            x[i][0]-=0.5
    x = x*scale   
    return x
    
    
def getRandomData1D(n,g,sig,grid1=False,grid2=False,relsize=0.3):
    #
    # returns n data points from 
    # a) a mixture of g Gaussians with covariance sig*unity (grid2=False)
    #   or
    # b) a mixture of g local clusters of data points with a square-like grid shape (grid2 = True)
    #
    # g cluster centers are taken from either
    # a) a uniform random distribution in the k-d unit square (grid=False)
    #   or
    # b) a square like grid with maximally square dimensions (really sqare if g is a square number)
    d=1            
    mean=0
    cov=sig # symmetric
    if grid1:
        centers = np.zeros([g,d]) # 2D array with 2nd dim=1
        centers = griddify1D(centers)
    else:
        centers = np.random.random([g,d])
        
                
    sizes=np.array([int(n/g)]*g)
    # fix last one
    sizes[-1]=n-sum(sizes[:g-1])
    X = None
    # make some clusters in the data set
    scale=0
    for i in range(g): 
        if grid2:
            newdat = np.zeros((sizes[i],d))
            scale = 1.0/(g)*relsize
            newdat = griddify1D(newdat,scale=scale,center=True)
        else:
            newdat = np.random.normal(mean, cov, sizes[i])

        cdata = centers[i]+ newdat
        if not X is None:
            X = np.concatenate((X,cdata))
        else:
            X = cdata
    # create 0 vector
    zeros=np.expand_dims(np.zeros(X.shape[0]),1)
    # concatenate
    X=np.concatenate((X,zeros),1)

    return X, centers, scale
