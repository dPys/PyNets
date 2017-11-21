import glob
import os
import pandas as pd
import numpy as np
from itertools import chain

###
working_path = r'/work/04171/dpisner/data/ABM/network_analysis' # use your path
name_of_network_pickle = 'net_metrics_sps'
net_name='CON'
missingness_thresh = 0.10
clusters=500
###

all_subs = [i for i in os.listdir(working_path) if len(i) <= 3]
all_subs.sort()
all_subs = all_subs[:-35]

for ID in all_subs:
    atlas_name =  ID + '_' + net_name + '_parcellation_' + str(clusters)
    subject_path = working_path + '/' + str(ID) + '/' + atlas_name
    ##Get path of thresholded pickles
    net_pickle_mt_list = []
    for i in os.listdir(subject_path):
        #        if name_of_network_pickle in i and net_name in i and ID in i and '.csv' not in i and '.txt' not in i and 'net_ts' not in i and '.graphml' not in i and '.nii.gz' not in i:
        if name_of_network_pickle in i and ID in i and 'mean' not in i and '.csv' not in i and '.txt' not in i and 'net_ts' not in i and '.graphml' not in i and '.nii.gz' not in i:
            try:
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

    list_ = list_[0:-1]
    try:
        ##Concatenate and find mean across dataframes
        df_concat = pd.concat(list_).mean().astype(str)
        print('Concatenating for ' + str(ID))
        df_concat['id'] = str(ID)
        df_concat = df_concat.to_frame().transpose()
        df_concat['id'] = df_concat['id'].astype(int)
#        df_concat.to_pickle(subject_path + '/' + str(ID) + '_' + name_of_network_pickle + '_' + net_name + '_mean')
        df_concat.to_pickle(subject_path + '/' + str(ID) + '_' + name_of_network_pickle + '_mean')
    except:
        print('NO OBJECTS TO CONCATENATE FOR ' + str(ID))
        continue

allFiles = []
for ID in all_subs:
    atlas_name =  ID + '_' + net_name + '_parcellation_' + str(clusters)
    #path_name = working_path + '/' + str(ID) + '/' + atlas_name + '/' + str(ID) + '_' + name_of_network_pickle + '_' + net_name + '_mean'
    path_name = working_path + '/' + str(ID) + '/' + atlas_name + '/' + str(ID) + '_' + name_of_network_pickle + '_mean'
    print(path_name)
    if os.path.isfile(path_name):
        print(path_name)
        allFiles.append(path_name)

allFiles.sort()

frame = pd.DataFrame()
list_ = []

for file_ in allFiles:
    try:
        df = pd.read_pickle(file_)
        df = df.drop(0, 1)
        new_names = [(i, net_name + '_' + i) for i in df.iloc[:, 1:].columns.values]
        df.rename(columns = dict(new_names), inplace=True)
        list_.append(df)
    except:
        print('File: ' + file_ + ' is corrupted!')
        continue

##Get only common columns
#common_cols = list(set.intersection(*(set(df.columns) for df in list_)))
#try:
#    common_cols.pop(0)
#except:
#    pass
#frame = pd.concat([df[common_cols] for df in list_], ignore_index=True, axis=0)

#frame = pd.concat(list_, axis=0, ignore_index=True)

list_of_dicts = [cur_df.T.to_dict().values() for cur_df in list_]
frame = pd.DataFrame(list(chain(*list_of_dicts)))

nas_list=[]
missing_thresh_perc=100*(missingness_thresh)
for column in frame:
    thresh=float(np.sum(frame[column].isnull().values))/len(frame)
    if thresh > missingness_thresh:
        nas_list.append(str(frame[column].name).encode().decode('utf-8'))
        print('Removing ' + str(frame[column].name.encode().decode('utf-8')) + ' due to ' + str(round(100*(frame[column].isnull().sum())/len(frame),1)) + '% missing data...')

##Remove variables that have too many NAs
for i in range(len(nas_list)):
    try:
        ##Delete those variables with high correlations
        frame.drop(nas_list[i], axis=0, inplace=True)
    except:
        pass

##Fix column order
frame = frame[frame.columns[::-1]]

##Replace zeroes with nan
try:
    frame[frame == 0] = np.nan
except:
    pass

out_path = working_path + '/' + name_of_network_pickle + '_' + net_name + '_' + str(clusters) + '_output.csv'
frame.to_csv(out_path, index=False)
