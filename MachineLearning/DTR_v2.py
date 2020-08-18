import numpy as np
import warnings
import sklearn
warnings.filterwarnings('ignore')

from .library import *
from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def desisionTreeRegression(df, features = None, target = None, percent_train = 0.8):

    X_train, y_train, X_test, y_test = get_X_y_from_df(df, features = features, target = target, percent_train = percent_train)

    pca = decomposition.PCA()
    decisionTree = tree.DecisionTreeRegressor()

    pipe = Pipeline(steps=[('pca', pca),
                           ('decisionTree', decisionTree)])

    n_components = list(np.arange(1, X_train.shape[1]+1))

    criterion = ['mse', 'friedman_mse', 'mae']
    max_depth = [3,5,8,12]

    parameters = dict(pca__n_components = n_components,
                      decisionTree__criterion = criterion,
                      decisionTree__max_depth = max_depth)

    reg = GridSearchCV(pipe, parameters)
    reg.fit(X_train, y_train)

    print('Best Criterion:', reg.best_estimator_.get_params()['decisionTree__criterion'])
    print('Best max_depth:', reg.best_estimator_.get_params()['decisionTree__max_depth'])
    print('Best Number Of Components:', reg.best_estimator_.get_params()['pca__n_components'])

    criterion = reg.best_estimator_.get_params()['decisionTree__criterion']
    max_depth = reg.best_estimator_.get_params()['decisionTree__max_depth']
    n_components =  reg.best_estimator_.get_params()['pca__n_components']

    pca = decomposition.PCA(n_components= n_components)
    decisionTree = tree.DecisionTreeRegressor(max_depth = max_depth, criterion= criterion)

    reg_op = make_pipeline(pca, decisionTree).fit(X_train,y_train)

    y_pred = reg_op.predict(X_test)

    error = mean_squared_error(y_test, y_pred, squared= False)
    return reg_op, error

#df = pd.read_csv('/home/ad/Downloads/LAB/DataMining_Lab/Filter_Methods/final.csv')
#reg, error = desisionTreeRegression(df, features= 1, target= 7, percent_train= 0.8)
#print(error)