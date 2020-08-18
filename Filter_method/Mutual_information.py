import sklearn
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from .library import *
#CALCULATE THE MUTUAL INFORMATION BETWEEN THE LAST COLUMN AND EACH LEFT FEATURES

#RETURN AN ARRAY OF MI SCORE FOR CONTINUOUS TARGET VARIABLE
def mutual_info_continuous(dataframe, target = None):
    n_feature = len(dataframe.columns) - 1
    if target == None:
        target = n_feature
    mi = mutual_info_regression(dataframe[dataframe.columns[:n_feature]],
                             dataframe[dataframe.columns[target]])
    return mi

#RETURN AN ARRAY OF MI SCORE FOR CATEGORICAL TARGET VARIABLE
def mutual_info_categorical(dataframe, target = None):
    n_feature = len(dataframe.columns) - 1
    if target == None:
        target = n_feature
    mi = mutual_info_classif(dataframe[dataframe.columns[:n_feature]],
                             dataframe[dataframe.columns[target]])
    return mi

def mutual_info(df, is_categories = True, col = -1, target = None):
    # RAISE EXCEPTION IF INPUT DATA IS NEITHER DATAFRAME NOR BUNCH
    if not isinstance(df, (pd.core.frame.DataFrame, sklearn.utils.Bunch)):
        raise Exception('Input data needs to be DATAFRAME or BUNCH')

    # CONVERT BUNCH TO DATAFRAME
    if isinstance(df, sklearn.utils.Bunch):
        df = convert_Bench2Dataframe(df)

    if target == None or target < 0 or target > len(df.columns) -1:
        target = len(df.columns)-1
    if is_categories:
        # CALCULATE SPECIFIC COLUMNS
        if (col != -1):
            print(f'Mutual information coefficient between: {df.columns[col]} and {df.columns[target]}')
            return mutual_info_classif(np.array(df[df.columns[col]]).reshape(-1,1), df[df.columns[target]])
        else:
            return mutual_info_categorical(df) #CALCULATE FOR ALL COLUMNS
    else:
        # CALCULATE SPECIFIC COLUMNS
        if (col != -1):
            print(f'Mutual information coefficient between: {df.columns[col]} and {df.columns[target]}')
            return mutual_info_regression(np.array(df[df.columns[col]]).reshape(-1,1), df[df.columns[target]])
        else:
            return mutual_info_continuous(df)  # CALCULATE FOR ALL COLUMNS

#df = pd.read_csv("/home/ad/Downloads/LAB/DataMining_Lab/crop_data_anva/crop_data.csv")
#print(mutual_info(df, False))

#from sklearn.datasets import load_iris
#data = load_iris()
#print(mutual_info(data, True, 2, target= 6))