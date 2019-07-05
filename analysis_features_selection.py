# importing python libraries
import pandas as pd
import os
import json
import numpy as np

#creating a list of lists of dictionaries 
directory = r"F:\Annotated_Trips"
dirListing = os.listdir(directory)
my_dict2=[]
for index, js in enumerate(dirListing):
    with open(os.path.join(directory,js),encoding='utf-8')as json_file:
        data = json.load(json_file)
        my_dict=data['samples']
        my_dict2.append(my_dict)
            json_file.write(json.dumps(data))
            
# just to make sure we dont have any files which doesnt have data
my_dict1=[]
for dict in my_dict2:
    if dict==None:
        continue
    my_dict1.append(dict)
    
#creating a list of dataframes to analyze the number of nans in every file
df_list2=[]
for dict in my_dict1:
    df=pd.DataFrame(dict)
    df_list2.append(df)
    
#removing cuted samples column from every datframe in the list as its not needed as it has all values null
df_cuted_samples=[]
for df in df_list2:
    for col in df.columns:
        if 'cuted_samples' in col:
            del df[col]
    
#to figure out what percentage of nan values are there in every column
df_quantify1=[]
for df in df_list2:
    df_quantify1.append(df.isnull().sum()/len(df))
    
#creating a list of percentage of nan values in every column and since we have made a datafrme of nan values so it will be max in that list
df_max=[]
for df in df_quantify:
    df_max.append(df.max())
    
#important libraries to find out index of files which contains more than a particular percentage of nan values
import collections
from collections import defaultdict

#finding indexes of all the values that were stored in df_max
d = defaultdict(list)   
for index, e in enumerate(df_quantify):
    d[e].append(index)
    
a=[]           
for key in d:
    a.append(d[key])
    
#removing the files from directorylisting that have more than 10% of the data missing
for i in range(3142):
    for x in a[i]:
        dirListing_new.append(dirListing[x])
        
#some files have only7 features in the starting which are creating a lot of nans and are also not the important features to us so before analyzing which features to take removing those trips
index2=[]           
for dict in my_dict1:
    index2.append([i for i, d in enumerate(dict) if len(d)==7]) 
    
#dataframe of diiferent number of features in allk files
df_quantify=[]
for df in df_list2:
    df_quantify.append(len(df.columns))
  
#finding indexes and hence creating dataframe with appropriate features
d = defaultdict(list)   
for index, e in enumerate(df_quantify):
    d[e].append(index)

    

