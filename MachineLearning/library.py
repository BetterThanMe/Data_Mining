import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

def convert_Bench2Dataframe(data):
    df = pd.DataFrame(data = np.c_[data['data'], data['target']],
                      columns= data['feature_names'] + ['diabetes'])
    return df
def get_X_y_from_df(df, features = None, target = None, percent_train = 0.8):

    if isinstance(df, sklearn.utils.Bunch):
        df = convert_Bench2Dataframe(df)

    if isinstance(features, int):
        features = np.array([features])

    X, y = [], []
    for col in features:
        if col < 0 or col > len(df.columns) - 1:
            raise Exception(f'The index feature {col} is out of range {len(df.columns)-1}')
        if isinstance(df[df.columns[col]][0], str):
            le = LabelEncoder().fit(df[df.columns[col]])
            df[df.columns[col]] = le.transform(df[df.columns[col]])
        X.append(df[df.columns[col]])
    X = np.array(X).T

    if target < 0 or target > len(df.columns) - 1:
        raise Exception(f'The index feature {target} is out of range {len(df.columns) - 1}')
    y = df[df.columns[target]]

    random_array =np.random.permutation(len(y))
    X_shuffered = np.zeros_like(X)
    y_shuffered = np.zeros_like(y)
    for i in range(len(y)):
        X_shuffered[i,:] = X[random_array[i], :]
        y_shuffered[i] = y[random_array[i]]

    num_train = int(percent_train*len(X))
    X_train, X_test = X_shuffered[:num_train], X_shuffered[num_train:]
    y_train, y_test = y_shuffered[:num_train], y_shuffered[num_train:]

    return X_train, y_train, X_test, y_test