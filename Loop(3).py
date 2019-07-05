df_list1_stats.index
import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
directory = r'F:\annotated_trips'
dirListing = os.listdir(directory)
i = y = 0
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
                df_final4 = df_global_data
                y += 1
            else:   
                df_final4 = df_final4.append(df_global_data,sort=True)
df_final4.to_csv('Final3.csv')
df_final4.to_msgpack('Final3.mpack', compress='zlib')
df_final_new=df_final3.append(df_final1)
plt.hist(x2, bins='auto')
x2=df_final4.isnull().sum()/len(df_final4)
df_final=pd.read_csv('Final2.csv')
df_final=df_final.drop('Unnamed: 0',axis=1)
df_final2=df_final.append(df_final1)
y=df_final2.isnull().sum()
x=df_final_new.isnull().sum()/len(df_final_new)
plt.hist(df, bins='auto')
plt.hist(x, bins='auto')
x1=df_final_new.isnull().sum()
null_percentage=df_final.isnull().sum()/len(df_final)
null_percentage.index
df_final.columns
a=df_final.isnull().sum()
a=df_final.isnull().sum()
df_final_new.index=range(3377)
df2=pd.read_csv('cenas1.csv')

df_final.to_csv('final.csv',sep=',')
df_final1=pd.read_csv('cenas.csv')
df_final1=df_final1.drop('Unamed: 0',axis=1)
a=df_final.isnull().sum()


df_new=df_list2[0].append(df_list2[1])
fig,ax=plt.subplots(figsize=(300,275))
null_percentage.plot(kind='bar')






