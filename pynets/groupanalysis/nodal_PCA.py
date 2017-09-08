#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 16:13:32 2017

@author: charleslaidi

This script can be used in the 'group analysis pipeline' of Pynets
There are two functions (so that the User can define or not the number of components)
"""

# There are two functions in this script

# Function A (pca_pynets) takes as an input a dataframe, performs the PCA on all components
# , produces a plot of the components (x) by explained variance (y), and then
# re execute the PCA with the number of components above the threshold
# which is defined by 1/(number of feature) (corresponding to an eigen value of 1)
# Function A returns a dataframe with the PCA components as columns and the subjects
# as a row. As explained here (https://stackoverflow.com/questions/42167907/understanding-scikitlearn-pca-transform-function-in-python)
# each value in the dataframe is the projection of each subject on a given component

# Functions B (pca_pynets_ncomp) is similar to function A, but one can specify the number of threshold
# that we want. The same plot as above is produced, and one can see 

# base must be a pandas datagrame
# eg: base = pandas.read_csv('/Users/charleslaidi/Desktop/charles.csv', index_col = 'id', sep = ",")
# path_output = '/Users/charleslaidi/Desktop' this is the output where the plot are produced 

# A. function with no custom number of components
def pca_pynets(base,path_output):
    # Essai pour faire une PCA 
    from sklearn.decomposition import PCA
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    
    # définition de la base en array à utiliser
    
    data=base.filter(regex='DMN*',axis=1) # name of the network 
    
    # ------------------------------------
    # transform the database into a matrix
    matrix_nodes = data.as_matrix()
    
    # ------------------------------------
    # PCA
    # ------------------------------------
    
    # def of PCA
    pca = PCA()
    
    # fit the PCA
    pca.fit(matrix_nodes)
    
    # apply the transform reduction to the dataset 
    # X_new = pca.transform(matrix_nodes) not used for this part 
    
    # explained variance
    variance = pca.explained_variance_ratio_ # gives the percentage of variance explained by each component
    
    # plot the components y = explained variance / x = components
    seuil = 1/float(matrix_nodes.shape[1])
    fig = plt.figure()
    horiz_line_data = np.array([seuil for i in xrange(len(variance))])
    liste = range(0,len(variance))
    plt.plot(liste, horiz_line_data, 'r--') 
    fig.suptitle('Explained variance X Components (thr = %.2f) \n thr = 1/(nb of nodes)'%(seuil), fontsize=10)
    plt.plot(title="Components by Explained variance")
    plt.ylabel('Percentage of explained variance')
    plt.xlabel('Components')
    plt.plot(liste,variance)
    fig.savefig('%s/Components by explained variance undefined threshold.png'%(path_output))
    #plt.show()
    plt.close()
    
    # calculating the number of remaining components after applying thresholds
    retained = variance > seuil
    
    # count the number of components > threshold
    nf = np.sum(retained)
    
    # rerunning the PCA with the number of components above the thr
    pca2 = PCA(n_components=nf)
    pca2.fit(matrix_nodes)
    X_new2 = pca2.transform(matrix_nodes)
    variance2 = pca2.explained_variance_ratio_
    liste2 = range(0,nf)
    liste3 = []
    for x in liste2: 
        h = x + 1
        z = 'Comp_n%s_expvar_%.2f_perc' %(h,variance2[x])
        liste3.append(z)
    df_final = pd.DataFrame(data=X_new2,columns=liste3)
    return df_final




# B. function with custom number of components
def pca_pynets_ncomp(base,path_output,n_comp=int):
    # Essai pour faire une PCA 
    from sklearn.decomposition import PCA
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    
    # définition de la base en array à utiliser
    
    data=base.filter(regex='DMN*',axis=1) # change this part with the name 
    
    # ------------------------------------
    # transform the database into a matrix
    matrix_nodes = data.as_matrix()
    
    # ------------------------------------
    # PCA
    # ------------------------------------
    
    # def of PCA
    pca = PCA(n_components=n_comp)
    
    # fit the PCA
    pca.fit(matrix_nodes)
    
    # apply the transform reduction to the dataset 
    X_new = pca.transform(matrix_nodes)
    
    # explained variance
    variance = pca.explained_variance_ratio_ # gives the percentage of variance explained by each component
    
    # plot the components y = explained variance / x = components
    seuil = 1/float(matrix_nodes.shape[1])
    fig = plt.figure()
    horiz_line_data = np.array([seuil for i in xrange(len(variance))])
    liste = range(0,len(variance))
    plt.plot(liste, horiz_line_data, 'r--') 
    fig.suptitle('Explained variance X Components custom(thr = %.2f) \n thr = 1/(nb of nodes)'%(seuil), fontsize=10)
    plt.plot(title="Components by Explained variance")
    plt.ylabel('Percentage of explained variance')
    plt.xlabel('Components')
    plt.plot(liste,variance)
    fig.savefig('%s/Components by explained variance custom threshold.png'%(path_output))
    #plt.show()
    plt.close()
    
    variance = pca.explained_variance_ratio_
    liste2 = range(0,n_comp)
    liste3 = []
    for x in liste2: 
        h = x + 1
        z = 'Comp_n%s_expvar_%.2f_perc' %(h,variance[x])
        liste3.append(z)
    df_final = pd.DataFrame(data=X_new,columns=liste3)
    #return X_new2
    return df_final

