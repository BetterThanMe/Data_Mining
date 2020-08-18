import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from .library import *
#features: array of index columns in dataframe to be the input data
#target: index of output columns
#percent_train: percantage of data train in dataset
def SVR_test(df, features = None, target = None, percent_train = 0.8, c = 1, gamma = 'scale', epsilon = 0.1, params = None):

    X_train, y_train, X_test, y_test = get_X_y_from_df(df, features = features, target = target, percent_train = percent_train)

    reg = make_pipeline(preprocessing.StandardScaler(),SVR(C= c, gamma= gamma, epsilon = epsilon))
    scores = np.zeros((2,))
    scores[0] = abs(np.mean(cross_val_score(reg, X_train, y_train, cv = 5, scoring= 'neg_root_mean_squared_error')))

    reg.fit(X_train, y_train)
    y_predict = reg.predict(X_test)
    scores[1] = mean_squared_error(y_test, y_predict, squared=False)

    return reg, scores

    #reg: MODEL PREDICT
    #scores[0]: mean RMSE of k - cross_validation, scores[1]: RMSE of test_data

#df = pd.read_csv('/home/ad/Downloads/LAB/DataMining_Lab/Filter_Methods/final.csv')
#print(df.columns)
#reg, scores = SVR_test(df, features=7, target= 8, c = 1, epsilon= 0.05)
#print(scores)