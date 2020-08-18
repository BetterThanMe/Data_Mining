from .library import *
from scipy.stats import f_oneway
import sklearn
#df = pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/difficile.csv")

df = pd.read_csv("/home/ad/Downloads/LAB/DataMining_Lab/Filter_Methods/final.csv")

#CREATE A DICTIONARY COMBINE A CATEGORICAL FEATURES AND A CONTINUOUS DEPENDENT VARIABLE
def create_anova_dic(array, target): #array: categorical feature; target: continuous variable
    a = np.array(array)
    group = np.unique(a)
    dic = dict()
    for i in group:
        dic[i] = []
    for i in range(len(array)):
        dic[array[i]].append(target[i])
    return dic

#CALCULATE ANOVA SCORE WITH A FEATURES CORRESPONDING WITH CONTINUOUS TARGET VARIABLE
def anova_1Features(df):
    dic = dict()
    if isinstance(df, sklearn.utils.Bunch):
        df = convert_Bench2Dataframe(df)
    if (isinstance(df, pd.DataFrame)):
        for attri in df.columns:
            dic[attri] = np.array(df[attri])
    elif (isinstance(df, dict)):
        dic = df
    else:
        raise Exception("Input data is not in the right form")

    List = []
    for i in dic:
        List.append(dic[i])
    stat, p = f_oneway(*List)
    return p

def anova_CategoricalFeature_Target(df, feature_idx, target_idx):
    if not is_categorical_columns(df[df.columns[feature_idx]]):
        raise Exception(f'{df.columns[feature_idx]} is not categorical')
    else:
        dic = create_anova_dic(df[df.columns[feature_idx]], df[df.columns[target_idx]])
        return anova_1Features(dic)

def anova(df, is_1Feature = False, features = None, targets = None):
    if is_1Feature == True:
        return anova_1Features(df)
    else:
        if isinstance(features, int):
            features = [features]
        if isinstance(targets, int):
            targets = [targets]
        if isinstance(features, (int, list, np.ndarray)):
            for col in features:
                if col <0 or col > len(df.columns) - 1:
                    raise Exception(f'Index of columns in features array out of range Columns {len(df.columns)}')
            for col in targets:
                if col <0 or col > len(df.columns) - 1:
                    raise Exception(f'Index of columns in targets array out of range Columns {len(df.columns)}')
            dic = dict()
            for target_idx in targets:
                for feature_idx in features:
                    dic[(df.columns[feature_idx], df.columns[target_idx])] = anova_CategoricalFeature_Target(df, feature_idx, target_idx)
            return dic
        else:
            raise Exception('Features or Targets have to be a number or integer array')

#print(df.columns)
#print(anova(df, features=[5,6], targets=7))

