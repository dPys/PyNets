import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#%% FUNCTIONS
def countMotifs(A,N=3):
    #
    #This can be considered as three stages:
    #(1) Enumerate subgraphs using method from
    #   "Efficient Detection of Network Motifs" (Wernicke,2006)
    #(2) Label subgraphs by isomorphic group
    #    For at least N=3,4 this works by identifying degree distribution
    #    but this should not work in general. Future development needs to
    #    generalize to higher values of N.
    #(3) Subgraph significance vs. random distributions is not currently
    #    implemented but should be in the future.
    import numpy as np
    from copy import copy
    
    assert N in [3,4], "Only motifs of size N=3,4 currently supported"
    
    X2=np.array([[k] for k in range(A.shape[0]-1)])
    for n in range(N-1):
        X=copy(X2)
        X2=[]
        for vsub in X:
            #find list of nodes neighboring vsub with a larger index than root v
            idx=np.where(np.any(A[(vsub[0]+1):,vsub],1))[0]+vsub[0]+1
            #only keep node indices not in vsub
            idx=idx[[k not in vsub for k in idx]]
            if len(idx)>0:
                #if new neighbors found, add all new vsubs to list
                X2.append([np.append(vsub,ik) for ik in idx])
        if len(X2)>0:
            X2=np.vstack(X2)
        else:
            raise Exception("No Fully Connected Subgraphs of Size {:}".format(N))
    X2=np.sort(X2,1)
    X2=X2[np.unique(np.ascontiguousarray(X2).view(np.dtype((np.void, X2.dtype.itemsize * X2.shape[1]))), return_index=True)[1]]
    from collections import Counter
    umotifs=Counter([''.join(np.sort(np.sum(A[x,:][:,x],1)).astype(int).astype(str)) for x in X2])
    return umotifs

def adaptiveThresh_4motifs(S,F,threshes=np.linspace(-0.8,1.5,201),plot_on=True,save_plot=False,plot_path='./adaptive_thresh.png'):
    # Inputs:
    # S = binary symmetric numpy array (adj. mat. of structural graph)
    # F = weighted symmetric numpy array (adj. mat. of functional graph)
    # threshes = 1d numpy array wth threshold values, as z-scores
    # Outputs:
    # ms = 4-motif counts in structural matrix
    # mfset = 4-motif counts in functional matrix at each threshold value in threshes
    assert np.all(np.unique(S)==np.array([0,1])),"Structural Matrix S must be binarized"
    assert np.all(S==S.T), "Structural Matrix A must be symmetric"
    assert np.all(F==F.T), "Functional Matrix F must be symmetric"
    
    #list of 
    m4lib=['1113','1122','1223','2222','2233','3333']
    #count 
    ms=countMotifs(S,N=4)
    ms=np.array([ms[k] for k in m4lib])
    def adaptiveThresh(a):
        thr=F.mean()+a*F.std()
        mf=countMotifs((F>thr).astype(int),N=4)
        mf=np.array([mf[k] for k in m4lib])
        return mf
    
    mfset=[adaptiveThresh(a) for a in threshes]
    mfset=np.vstack(mfset)

    if plot_on:
        vmax=np.maximum(mfset.max(),ms.max())
        fig=plt.figure()
        plt.get_current_fig_manager().window.setGeometry(257,29,891,652)
        ax=plt.subplot2grid((3,4),(0,0),colspan=3)
        plt.imshow(ms[None,:],aspect='auto',vmin=0,vmax=vmax)
        plt.xticks([]);plt.yticks([])
        plt.ylabel('Structural')
        
        
        plt.subplot2grid((3,4),(1,0),rowspan=2,colspan=3)
        im=plt.imshow(mfset,aspect='auto',vmin=0,vmax=vmax)
        plt.xticks(range(len(m4lib)),m4lib)
        plt.xlabel('4-Node Motif')
        plt.ylabel('Functional Network\n(z-score threshold)')
        ylim=plt.ylim()
        yt=plt.yticks()[0]
        yt=yt[(yt>=0)&(yt<len(threshes))].astype(int)
        plt.yticks(yt,['{:0.1f}'.format(threshes[y]) for y in yt])
        plt.ylim(ylim)
        plt.gca().invert_yaxis()
        
        adist=np.sum(np.abs(mfset-ms),1)
        othresh=threshes[np.argmin(adist)]
        plt.subplot2grid((3,4),(1,3),rowspan=2)
        plt.plot(adist,threshes)
        plt.ylim([threshes.min(),threshes.max()])
        plt.xlabel('Sum Abs. Diff.\nOf Motif Counts')
        plt.yticks([])
        xl=plt.xlim()
        plt.plot(xl,[othresh,othresh],':')
        plt.xlim(xl)
        plt.gca().set_title('Opt. Thresh = {:0.2f}'.format(othresh),fontsize=10)
        
        cax=fig.add_axes(ax.get_position().from_bounds(0.71,0.66,0.02,0.21))
        fig.colorbar(im, cax=cax, orientation="vertical",label='Motif Counts')
        
        if save_plot:
            plt.savefig(plot_path)
    return ms,mfset
