import glob
import os
import pandas as pd
import numpy as np

###
working_path = r'/work/04171/dpisner/data/ABM/network_analysis' # use your path
name_of_network_pickle = 'DMN_net_mets_sps_cov'
missingness_thresh = 0.10
#atlas_name='Power 2011 atlas'
atlas_name='Dosenbach 2010 atlas'
###

all_subs = [i for i in os.listdir(working_path) if len(i) <= 3]
all_subs.sort()

for ID in all_subs:
    subject_path = working_path + '/' + str(ID) + '/' + atlas_name
    ##Get path of thresholded pickles
    net_pickle_mt_list = []
    for i in os.listdir(subject_path):
        if name_of_network_pickle in i and '.csv' not in i:
            try:
                val = int(i.split('_0.')[1])
                if val < 100 and val >= 90:
                    net_pickle_mt_list.append(i)
            except:
                continue

    net_pickle_mt_list.sort()

    allFiles = []
    for net_pick in net_pickle_mt_list:
        path_name = working_path + '/' + str(ID) + '/' + atlas_name + '/' + net_pick
        if os.path.isfile(path_name):
            allFiles.append(path_name)

    frame = pd.DataFrame()
    list_ = []

    for file_ in allFiles:
        df = pd.read_pickle(file_)
        list_.append(df)

    try:
        ##Concatenate and find mean across dataframes
        print('Concatenating for ' + str(ID))
        df_concat = pd.concat(list_).mean().astype(str)
        df_concat['id'] = str(ID)
        df_concat = df_concat.to_frame().transpose()
        df_concat['id'] = df_concat['id'].astype(int)
        df_concat.to_pickle(subject_path + '/' + str(ID) + '_' + name_of_network_pickle + '_mean')
    except:
        print('NO OBJECTS TO CONCATENATE FOR ' + str(ID))
        continue

allFiles = []
for ID in os.listdir(working_path):
    path_name = working_path + '/' + str(ID) + '/' + atlas_name + '/' + str(ID) + '_' + name_of_network_pickle + '_mean'
    if os.path.isfile(path_name):
        print(path_name)
        allFiles.append(path_name)

allFiles.sort()

frame = pd.DataFrame()
list_ = []

for file_ in allFiles:
    try:
        df = pd.read_pickle(file_)
        list_.append(df)
    except:
        print('File: ' + file_ + ' is corrupted!')
        continue

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

out_path = working_path + '/' + name_of_network_pickle + '_' + atlas_name  + '_output.csv'
frame.to_csv(out_path, index=False)
