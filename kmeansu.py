from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import bottleneck as bn
import copy
import math
import random
from bfutil import errorOf
from collections import Counter

#random vector from surface of d-dimensional unit hypersphere
def randUnitVec(d):
    inv_d = 1.0 / d
    gauss = np.random.normal(size=d)
    length = np.linalg.norm(gauss)
    if length == 0.0:
        x = gauss
    else:
        r = np.random.rand() ** inv_d
        x = np.multiply(gauss, r / length)
    length = np.linalg.norm(x)
    x/=length
    return(x)

#
# the k-means-u and k-means-u* algo 
#
# X:data set
# kmeans: container object for
# - cluster_centers_
# - k (implicitly as len(cluster_centers))
# - inertia
# - store - keep history of codebooks
# - maxRetry=0: k-means-u,  maxRetry >0: k-means-u*
#
# returns: kmeans object
#
def kmeansU(X,kmeans, store=False, maxRetry=0,ofac=0.01, loud=False):
    best = kmeans.inertia_ # lowest SSE so far
    Cbest = copy.deepcopy(kmeans.cluster_centers_)
    k = len(kmeans.cluster_centers_)
    retry = 0
    successfulRetries = 0
    uhist = {}# stores quadruples of codebook, i_maxerr,i_minutil, boolean regular (indicates if no retrial)
    n_iter=0 # number of Lloyd iterations
    uruns=0  # number of non-local jumps
    noRetryErr=-1;
    if maxRetry > 0:
        algo = "kms"
    else:
        algo = "kmu"
    while True: # no repeat until in python, emulated as while True + break
        while True:
            cnt=Counter()
            C = kmeans.cluster_centers_ # current codebook, e.g. from random or from kmeans

            # compute error and utility for all centers
            error = np.zeros(k)
            utility = np.zeros(k)
            for i,x in enumerate(X): #loop over all data points     
                delta = C-x # substract current data point from centers                
                dis = np.linalg.norm(delta,axis=1) # compute L2 norm
                # find (indices of) the two closest centers (bmu=best matching unit) for the current data point
                bmu1i, bmu2i = sorted(np.argpartition(dis,2)[:2],key=lambda x: dis[x])
                cnt[bmu1i]+=1 # count for how much signals each center is bmu
                e1=dis[bmu1i]**2      # squared distance to bmu1
                e2=dis[bmu2i]**2      # squared distance to bmu2
                error[bmu1i]+=e1      # accumulate error
                utility[bmu1i]+=e2-e1 # accumulate utility
                
            i_maxerr = np.argmax(error) # determine index of center with max err
            i_minutil = np.argmin(utility) # determine index of center min utility

            # handle special case that mu and lambda are identical
            if i_maxerr == i_minutil:
                if loud:
                    print("same same",i_maxerr)
                break
            uruns+=1 # count the jumps

            if store: # just for producing step-by-step figures, not needed for k-means-u
                # store current codebook (before jump)!!!!)
                cc=copy.deepcopy(C)
                #print("store",i_maxerr,i_minutil,True, len(uhist))
                uhist[len(uhist)]={"codebook":cc,"i_mu":i_maxerr,"i_lambda":i_minutil,"regular":True,"jumps":uruns, "algo":algo}
               
            # JUMP
            md = math.sqrt(error[i_maxerr]/cnt[i_maxerr]) # standard dev around max err unit
            offset=randUnitVec(X.shape[1])*md*ofac # offset vector
            C[i_minutil]=C[i_maxerr]+offset # re-position $\lambda$
            C[i_maxerr]-=offset             # re-position $\mu$

            # REGULAR K-MEANS
            kmeans = KMeans(n_clusters=k, init = C)
            kmeans.fit(X)
            # count overall Lloyd iterations
            n_iter += kmeans.n_iter_

            if kmeans.inertia_ < best:
                # improvement!   ==> continue ....
                best = kmeans.inertia_
                Cbest = copy.deepcopy(kmeans.cluster_centers_)
                if retry>0:
                    successfulRetries += 1
                    if loud:
                        print("successful retry:",retry,best)
                retry=0 # reset retry counter
            else:
                # no improvement ==> terminate inner loop
                break
            # **************** end of inner loop *****************

        retry+=1
        # store best error of normal k-means-u (without retry)
        if retry == 1 and noRetryErr < 0:
            # remember the error w/o retry (i.e. the k-means-u error
            noRetryErr = best
            # iterations of k-means-u
            n_iter_u0=n_iter
       # keep history if requested
        if store: 
            # store this(worsened) codebook
            cc=copy.deepcopy(kmeans.cluster_centers_)
            #print("store",0,0,False, len(uhist))
            uhist[len(uhist)]={"codebook":cc,"i_mu":0,"i_lambda":0,"regular":False,"jumps":uruns, "algo":algo}

            #uhist[len(uhist)]=cc,0,0,False,uruns
        # rewind to best so far
        kmeans.cluster_centers_ = copy.deepcopy(Cbest)
        kmeans.successfulRetries_ = successfulRetries
        if retry > maxRetry:
            break # finish!
        else:
            if loud:
                print("retry #",retry)
        # **************** end of outer loop *****************
   
    # kmeans as return container object for
    #  - cluster_centers_ of best solution
    #  - k (implicitely as len(cluster_centers))
    kmeans.n_iter_un=n_iter
    kmeans.n_iter_u0=n_iter_u0
    kmeans.uruns = uruns
    kmeans.noRetryErr = noRetryErr
    
    kmeans.cluster_centers_ = copy.deepcopy(Cbest)
    kmeans.inertia_=best
    kmeans.uhist=uhist
    return kmeans

# map codebook to clusters
# - g = number of clusters in the data set
# - centers = cluster centers in the data set
# - codebook = k centers positioned by some clustering method
# - ratio=(# of codebook vectors)/g
#
# returns: returns vector of range g with number of vectors assoziated with cluster g,
# but centered to optimal value (by substracting ratio) and shifted by 50 to be in the positive range
#
# So an optimal solution would return a vector with all elements equal to 50

def getMap(g,centers,codebook,ratio):
    # which k-means center is mapped to which distribution cluster
    kk = KMeans(n_clusters=g, init=centers)
    kk.fit(centers)

    # Counter 0 - map kmeans codebook to PD clusters - a PD cluster may be not present
    pred0 = kk.predict(codebook)
    a = Counter(pred0) # contains info for each distribution cluster, how many vectors are mapped on it
    #print(a)
    for i in range(g):
        if not i in a:
            a[i]=0   
    cmap=np.array([a[i] for i in sorted(a.keys())])-ratio+50
    return cmap
