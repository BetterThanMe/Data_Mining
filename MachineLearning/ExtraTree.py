from .library import *
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def ExtraTreeReg(df, features = None, target = None, percent_train = 0.8):

    X_train, y_train, X_test, y_test = get_X_y_from_df(df, features= features, target = target, percent_train = percent_train)

    extraTree = tree.ExtraTreeRegressor()

    pipe = Pipeline(steps=[('extraTree', extraTree)])

    max_features = ['sqrt', 'log2', 2, 0.2]
    min_samples_split = [8,10,11,13, 20]
    criterion = ['mse', 'friedman_mse', 'mae']

    parameters = dict(extraTree__criterion = criterion,
                      extraTree__max_features = max_features,
                      extraTree__min_samples_split = min_samples_split)

    reg = GridSearchCV(pipe, parameters)
    reg.fit(X_train, y_train)

    print('Best criterion : ', reg.best_estimator_.get_params()['extraTree__criterion'])
    print('Best max_features = ', reg.best_estimator_.get_params()['extraTree__max_features'])
    print('Best min_sample_split = ', reg.best_estimator_.get_params()['extraTree__min_samples_split'])

    criterion = reg.best_estimator_.get_params()['extraTree__criterion']
    max_features = reg.best_estimator_.get_params()['extraTree__max_features']
    min_samples_split = reg.best_estimator_.get_params()['extraTree__min_samples_split']

    reg_op = tree.ExtraTreeRegressor(criterion = criterion, max_features= max_features, min_samples_split= min_samples_split)
    reg_op.fit(X_train, y_train)

    y_pred = reg_op.predict(X_test)
    score  = mean_squared_error(y_test, y_pred, squared= False)

    return reg_op, score

#df = datasets.load_diabetes()
#_, score = ExtraTreeReg(df, features= np.arange(0,10), target= 10, percent_train= 0.8)
#print(score)

