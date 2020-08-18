from .library import *
from .SVR import *
from .ExtraTree import *
from .DTR_v2 import *
from .VotingTree import *
from sklearn import datasets

class ML:
    def __init__(self, df):
        self.dataframe = df
        print('Contains method: \n\
               _SVR: Support Vector Regression\n\
               _DecisionTreeRegression\n\
               _ExtraTreeRegression\n\
               _VotingTreeRegression')
    def __getitem__(self):
        return self.dataframe
    #Return a class model and score of testing
    def _SVR(self, features = None, target = None, percent_train = 0.8, c = 1, gamma = 'scale', epsilon = 0.1, params = None):
        return SVR_test(self.dataframe, features= features, target= target, percent_train=percent_train,
                        c=c, gamma= gamma, epsilon = epsilon, params=params)
    def _DecisionTreeRegression(self, features = None, target= None, percent_train= .8):
        return desisionTreeRegression(self.dataframe, features= features, target= target, percent_train= percent_train)
    def _ExtraTreeRegression(self, features = None, target= None, percent_train = .8):
        return ExtraTreeReg(self.dataframe, features= features, target= target, percent_train= percent_train)

    #Return class model, not contain score test
    def _VotingTreeRegression(self, features = None, target= None, percent_train = .8):
        return VotingTree(self.dataframe, features= features, target= target, percent_train= percent_train)

#df = datasets.load_diabetes()
#ml = ML(df)
#reg = ml._VotingTreeRegression(features= [0,1,2,9], target= 10)

