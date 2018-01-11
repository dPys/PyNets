import glob
import os
import pandas as pd
###
working_path = r'/work/04171/dpisner/data/ABM/network_analysis/' # use your path
name_of_network_pickle = 'net_metrics_sps_mean'
###

allFiles = []
for ID in os.listdir(working_path):
    path_name = working_path + ID + '/' + ID + '_' + name_of_network_pickle
    if os.path.isfile(path_name):
        print(path_name)
        allFiles.append(path_name)

frame = pd.DataFrame()
list_ = []

for file_ in allFiles:
    df = pd.read_pickle(file_)
    list_.append(df)

frame = pd.concat(list_)

out_path = working_path + '/' + name_of_network_pickle + '_output.csv'
frame.to_csv(out_path)
