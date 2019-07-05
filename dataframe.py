
import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
directory = r'F:\Annotated_Trips'
dirListing = os.listdir(directory)
i = y = 0
#going through every file
for filename in dirListing_new:
    i += 1
    print(round(i / len(dirListing_new) * 100, 2), '%')
    if filename.endswith(".json"):
        with open(os.path.join(directory, filename)) as json_file:
            j = 0
            data1 = json.load(json_file)
            my_dict = data1['global_data']
            df_global_data = pd.DataFrame.from_dict(my_dict, orient='index')
            df_global_data = df_global_data.transpose()
            my_dict2 = data1['samples']
            if my_dict2==None:
                continue
            for sample in my_dict2:
                df_single_sample = pd.DataFrame.from_dict(sample, orient='index')
                df_single_sample = df_single_sample.transpose()
                if j == 0:
                    df_samples = df_single_sample
                    j += 1
                else:
                    df_samples = df_samples.append(df_single_sample, sort=True)
            if 'cuted_samples' in df_samples:
                df_samples = df_samples.drop(columns='cuted_samples')
            df_samples_stats = df_samples.describe(include='all')
            for index1, row in df_samples_stats.T.iteritems():
                for column_name in list(df_samples_stats):
                    new_column_name = column_name + '_' + index1
                    df_global_data.loc[:,new_column_name] = row[column_name]
            if y == 0:
                df_final4 = df_global_data
                y += 1
            else:   
                df_final4 = df_final4.append(df_global_data,sort=True)
df_final4.to_csv('Final3.csv')
df_final4.to_msgpack('Final3.mpack', compress='zlib')
                
            for sample in my_dict2[len(index1):]:
                df_single_sample = pd.DataFrame.from_dict(sample, orient='index')
                df_single_sample = df_single_sample.transpose()
                if j == 0:
                    df_samples = df_single_sample
                    j += 1
                else:
                    df_samples = df_samples.append(df_single_sample, sort=True)
            if 'cuted_samples' in df_samples:
                df_samples = df_samples.drop(columns='cuted_samples')
            df_samples_stats = df_samples.describe(include='all')
            for index1, row in df_samples_stats.T.iteritems():
                for column_name in list(df_samples_stats):
                    new_column_name = column_name + '_' + index1
                    df_global_data.loc[:,new_column_name] = row[column_name]
            if y == 0:
                df = df_global_data
                y += 1
            else:   
                df_final4 = df_final4.append(df_global_data,sort=True)
#saving the final dataframe
df.to_csv('Final3.csv')
df.to_msgpack('Final3.mpack', compress='zlib')






