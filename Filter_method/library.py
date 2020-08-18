import pandas as pd
from sklearn.datasets import load_diabetes
import numpy as np

#OR ANY AVAILABLE DATAFRAME
def convert_Bench2Dataframe(data):
    df = pd.DataFrame(data = np.c_[data['data'], data['target']],
                      columns= data['feature_names'] + ['diabetes'])
    return df

#GET ALL COMBINATION FROM A ARRAY
def combination(list):
    n = len(list)
    array = np.empty((1,2))
    for i in range(n):
        j = i+1
        while j<n:
            array = np.append(array, np.array([list[i], list[j]]).reshape(1,-1), axis= 0)
            j+=1
    return array[1:,:]

#DESCENDING RANKING CONTINUOUS VALUE IF THE INPUT ARRAY WAS NOT RANKED
def ranking_continuous_value_array(array):
    temp = (-array).argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))+1
    return ranks

#CONVERT UNRANKED DATAFRAME TO RANKED DATAFRAME
def convert_continuousDf_rankingDf(data):
    df = pd.DataFrame.copy(data, deep=True)
    for i in df.columns:
        df[i] = ranking_continuous_value_array(data[i])
    return df

#CREATE CONTIGENCY TABLE FROM 2 CATEGORY VARIABLE WITH 2 NAME OF 2 FEATURES(COLUMNS NAME) OF A DATAFRAME
def create_contigency_from_2CA(df, f1, f2):
    count = np.zeros((len(df[df.columns[f1]].unique()), len(df[df.columns[f2]].unique())))
    for ix, i in enumerate(df[df.columns[f1]].unique(),start=0):
        for jx, j in enumerate(df[df.columns[f2]].unique(), start= 0):
            count[ix,jx] = 0
            for x in range(len(df)):
                if (df[df.columns[f1]][x] == i and df[df.columns[f2]][x]== j):
                    count[ix,jx]+=1
    return count

#Check if a array is cagorical or continuous
def is_categorical_columns(arr):
    a = np.array(arr)
    uniq_value = np.unique(a)
    if len(uniq_value)/len(a) <= 0.1:
        return True
    else:
        return False
