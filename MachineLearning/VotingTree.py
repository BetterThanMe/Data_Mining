from .DTR_v2 import *
from .ExtraTree import *
from .library import *
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import FeatureUnion
from sklearn import datasets

class VotingTree:
    def __init__(self, df, features = None, target = None, percent_train = 0.8):
        X_train, y_train, X_test, y_test = get_X_y_from_df(df, features=features, target=target, percent_train=percent_train)

        self.DTR_reg, DTR_score = desisionTreeRegression(df, features=features, target=target, percent_train=percent_train)
        self.Extra_reg, Extra_score = ExtraTreeReg(df, features=features, target=target, percent_train=percent_train)

        y_pred_DTR = self.DTR_reg.predict(X_test)
        y_pred_Extra = self.Extra_reg.predict(X_test)
        y_pred = (y_pred_DTR+y_pred_Extra)/2

        score = mean_squared_error(y_test, y_pred, squared= False)

        print(f'DecisionTree: {DTR_score} -- ExtraTree: {Extra_score} -- VotingTree: {score}')

        min_score = min(DTR_score, Extra_score, score)
        if min_score == score:
            self.model = [self.DTR_reg, self.Extra_reg]
        if min_score == DTR_score:
            self.model = [self.DTR_reg]
        if min_score == Extra_score:
            self.model == [self.Extra_reg]

    def predict(self, X_test):
        y_preds = []
        for model in self.model:
            y_preds.append(model.predict(X_test))

        y_total_pred = np.zeros_like(y_preds[0])
        for y_pred in y_preds:
            y_total_pred += y_pred
        y_total_pred /= len(y_preds)
        return y_total_pred

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return mean_squared_error(y_test, y_pred, squared= False)

    def __getitem__(self):
        return str(self.model)
    def net(self):
        name = ''
        for model in self.model:
            name += str(model) +'\n'
        return name
#df = datasets.load_diabetes()
#VT = VotingTree(df, features= np.arange(0,10), target= 10, percent_train= 0.8)
#print(VT.net())