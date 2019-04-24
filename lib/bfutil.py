import os
import matplotlib.patches as patches
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc



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

# in a list ox axes set all dims to that of the first axis
# or to the dimensions given in the parameters
def eqlim(axs,xlim=None,ylim=None,unitpad=None):
    try:
        axs=axs.flat
    except AttributeError:
        #print("attri .....")
        axs=[axs]
        pass
    #assert len(axs)>1
    a0=axs[0]
    if unitpad is not None:
        # take unitsquare with padding of unitpad
        xlim = (-unitpad,1+unitpad)
        ylim = xlim
    else:
        # take limits of first ax if not specified otherwise
        if xlim is None:
            xlim=a0.get_xlim()
        if ylim is None:
            ylim=a0.get_ylim()
    # give same limits to all axes
    for ax in axs:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
# compute squared error of given codebook and given dataset
def errorOf(C,X):
    mat = np.zeros([X.shape[0],C.shape[0]])
    for i in range(len(C)):
        #di = (X-C[i])
        mat[:,i] = (np.linalg.norm(X-C[i],axis=1))
    return sum(np.min(mat,axis=1)**2)

# modify graph o have no axis ticks
def noticks(ax):
    ax.tick_params(length=0)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
import inspect

# returns the name of the current function
def whoami():
    return inspect.stack()[1][3]

# finishes a plot, possibly writes it to file
def finish(fig,name, top=0.95, write=False, tight=True, dpi=100, imgdir='img', **kwargs):
        if tight:
            fig.tight_layout()
            plt.subplots_adjust(top=top)     # Add space at top

        if len(name)>4 and name[:4]=="fig_":
            name = name[4:]
        fname = os.path.join(imgdir,name+".png")
        if write:
            fig.savefig(fname,bbox_inches='tight', dpi=dpi, **kwargs)
            print("wrote:", fname)
        else:
            pass
        return(fig)

# prepares a multiplot
def makeplot(lines,cols,figsize=(20,6),title=None):
    f,axs = plt.subplots(lines, cols, figsize=figsize)
    if title is not None:
        f.suptitle(title, fontsize=16)
    return f,axs

# general plotting function for one kmeansX result    
def plot3(ax,
          X=None, # data set
          C=None, # codebook
          cap=None, # caption
          centers=[], # cluster centers
          sig=0.1, # sigma for Gaussians
          cmap=None, # colormap
          boxlen=0, # sidelength of color boxes
          ticks=False, # have ticks?
          fontsize=16, # fontsize for caption
          dotsize=2, # size of data points
          ypos=1.02, # position for caption
          vbar=False, # draw vertical bars for each codebook vector (useful for 1D)
          voro=False): # voronoi diagram
    
    cocy = ['green', 'red', 'yellow']

    # Voronoi diagram wrt centers
    if voro and C is not None and C.shape[0]>2:
        vor = Voronoi(C)
        voronoi_plot_2d(vor,ax=ax, show_vertices=False, show_points=True,line_colors="blue")
        
    # data
    if X is not None:
        ax.set_color_cycle(cocy);
        ax.plot(X[:,0],X[:,1],".", ms=dotsize);
    
    # codebook
    if C is not None:
        ax.plot(C[:,0],C[:,1],"o", ms=3,color="red");
        if vbar:
            o=np.array((0,0.1))
            lines = [[c-o,c+o] for c in C]
            lc = mc.LineCollection(lines, colors="red", linewidths=2,zorder=10)
            ax.add_collection(lc)
    
    # marker rectangles
    if cmap is not None:
        # marker rectangles
        if boxlen > 0:
            side = boxlen # boxlen given, use it
        else:
            side=max(sig*150,0.05) # derive from sig with lower bound
        col = ["orange"]*49+["red","green","blue"]+["#00CED1"]*49
        #col = ["yellow","red","orange","green","blue","violet","cyan","yellow"]+["brown"]*30
        # paint squares depending on numbers per center
        for i,x in enumerate(centers):
            if cmap[i] != 50: # 50 is the "optimal value ......, i.e. exactly the value of ratio
                lwidth= 2
            else:
                lwidth = 3
            ax.add_patch(
                patches.Rectangle(
                    (x[0]-side/2, x[1]-side/2),
                    side,
                    side,
                    edgecolor=col[cmap[i]],
                    linewidth=lwidth,
                    fill=False      # remove background
                )
            )
    
    # ticks
    if not ticks:
        noticks(ax)
        
    # caption
    if cap is not None:
        plt.text(0.5, ypos, cap,
        horizontalalignment='center',
        fontsize=fontsize,
        transform = ax.transAxes)


#visualizing high-D dataset by plotting all pairs of coordinate        
def dplot1(X,ticklabels=True,ticks=True,shareaxes=False,size=20):
    #X = X[:200]
    d = X.shape[1]
    fig,axs = plt.subplots(d,d, sharex=shareaxes, sharey=shareaxes)
    fig.set_size_inches(size,size)
    for y in range(d):
        for x in range(d):
            ax = axs[y,x]
            if x == y:
                # mark background on diagonal for orientation
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(1.5)
                #ax.set_axis_bgcolor((0.9,0.9,0.9))
            ax.plot(X[:,x],X[:,y],".", ms=0.5,fillstyle="full");
            #ax.plot(X[:,x],X[:,y],".", ms=5.5,fillstyle="full");
            if not ticks:
                ax.tick_params(length=0)
                ticklabels=False
            if not ticklabels:
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])

    fig.tight_layout()
    return fig

def dplot(X,ticklabels=True,ticks=True,shareaxes=False,size=20,ms=0.5):
    d = X.shape[1] # dimension of data
    if d==2:
        fig,ax = plt.subplots(1,1, sharex=shareaxes, sharey=shareaxes)
        ax.plot(X[:,0],X[:,1],"g.", ms=ms,fillstyle="full");
        if not ticks:
            ax.tick_params(length=0)
            ticklabels=False
        if not ticklabels:
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])       
    else:
        fig,axs = plt.subplots(d,d, sharex=shareaxes, sharey=shareaxes)
        for y in range(d):
            for x in range(d):
                ax = axs[y,x]
                ax.plot(X[:,x],X[:,y],"g.", ms=ms,fillstyle="full");
                if not ticks:
                    ax.tick_params(length=0)
                    ticklabels=False
                if not ticklabels:
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])

    # make distance between subplots really small
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.set_size_inches(size,size)
    return fig