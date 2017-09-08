import glob
import os
import pandas as pd
import numpy as np
###
#working_path = r'/work/04171/dpisner/data/ABM/network_analysis' # use your path
name_of_network_pickle = 'CON_SN_PREC_net_mets_sps_cov'
missingness_thresh = 0.10
atlas_name='Power 2011 atlas'
working_path = r'/work/04171/dpisner/data/ABM/network_analysis/002/Power 2011 atlas' # use your path

#atlas_name='Dosenbach 2010 atlas'
###

allFiles = []
for ID in os.listdir(working_path):
    path_name = working_path + ID + '/' + atlas_name + '/' + ID + '_' + name_of_network_pickle + '_' + ID
    if os.path.isfile(path_name):
        print(path_name)
        allFiles.append(path_name)

allFiles.sort()

frame = pd.DataFrame()
list_ = []

for file_ in allFiles:
    df = pd.read_pickle(file_)
    list_.append(df)

frame = pd.concat(list_, axis=0)

nas_list=[]
missing_thresh_perc=100*(missingness_thresh)
for column in frame:
    thresh=float(frame[column].isnull().sum())/len(frame)
    if thresh > missingness_thresh:
        nas_list.append(frame[column].name.encode('ascii'))
        print('Removing ' + str(frame[column].name.encode('ascii')) + ' due to ' + str(round(100*(frame[column].isnull().sum())/len(frame),1)) + '% missing data...')

##Remove variables that have too many NAs
for i in range(len(nas_list)):
    ##Delete those variables with high correlations
    frame.drop(nas_list[i], axis=1, inplace=True)

##Fix column order
frame = frame[frame.columns[::-1]]

##Replace zeroes with nan
frame[frame == 0] = np.nan

out_path = working_path + '/' + name_of_network_pickle + '_output.csv'
frame.to_csv(out_path, index=False)
