import numpy as np
from pynets.netmotifs import countMotifs,adaptiveThresh_4motifs
from pathlib import Path

try:
    import cPickle
except ImportError:
    import _pickle as cPickle

def motifCount_test():
    #loads a collection of adjacency matrices along with the motif-counts
    #calculated using package graph-tool
    #and asserts that our implementation should deliver same results
    #for counting motifs of size 3, 4
    gt_ref = Path(__file__).parent/"examples"/"motif_examples"/"graph_tool_results.pickle"
    with open(gt_ref,'rb') as fhandle:
        d=cPickle.load(fhandle)
        motif3=d['motif3']
        motif4=d['motif4']
        Aset=d['Aset']
    m3lib=['112','222']
    m4lib=['1113','1122','1223','2222','2233','3333']
    M3=[]
    M4=[]
    for k in range(Aset.shape[2]):
        A=Aset[:,:,k]
        m3=countMotifs(A,N=3)
        m3=np.sort([m3[k] for k in m3lib])
        m4=countMotifs(A,N=4)
        m4=np.sort([m4[k] for k in m4lib])
        M3.append(m3)
        M4.append(m4)
    M3=np.vstack(M3)
    M4=np.vstack(M4)
    assert np.all(M3==motif3), "Counts of N=3 motifs don't match graph-tool results"
    assert np.all(M4==motif4), "Counts of N=4 motifs don't match graph-tool results"

def adaptiveThresh_test():
    Spath=Path(__file__).parent/"examples"/"motif_examples"/'sub002_CON_structural_mx.txt'
    Fpath=Path(__file__).parent/"examples"/"motif_examples"/'sub002_CON_func_mx.txt'
    S=np.genfromtxt(Spath)
    F=np.genfromtxt(Fpath)
    n=F.shape[0]
    F[range(n),range(n)]=0
    S=np.maximum(S,S.T)
    F=np.maximum(F,F.T)
    S=(S!=0).astype(int)
    ms,mfset=adaptiveThresh_4motifs(S,F,threshes=np.linspace(-0.8,1.5,21),plot_on=False)
    
    assert len(ms)==6,'Structural 4-Motifs did not return array of length 6'
    assert mfset.shape==(21,6),"Functional Thresholded 4-Motifs did not generate (21,6) array"
