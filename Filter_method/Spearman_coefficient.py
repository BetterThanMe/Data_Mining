import sklearn
from .library import *
from scipy.stats import spearmanr

#CALCULATE SPEARMAN COEFFICIENT OF RANKED DATAFRAME
def spearman(df, col1 = -1, col2 = -1, ranked = False):

    # RAISE EXCEPTION IF INPUT DATA IS NEITHER DATAFRAME NOR BUNCH
    if not isinstance(df, (pd.core.frame.DataFrame, sklearn.utils.Bunch)):
        raise Exception('Input data needs to be DATAFRAME or BUNCH')

    #CONVERT BUNCH TO DATAFRAME
    if isinstance(df, sklearn.utils.Bunch):
        df = convert_Bench2Dataframe(df)

    #CONVERT CONTINUOUS DATAFRAME TO RANKED DATAFRAME
    if ranked == False:
        df = convert_continuousDf_rankingDf(df)

    #CALCULATE SPECIFIC COLUMNS
    if (col1 != -1 and col2 != -1):
        return spearmanr(df[df.columns[col1]], df[df.columns[col2]])[0]
    #CALCULATE ALL COLUMNS
    coef = dict()
    for couple in combination(df.columns):
        coef[(couple[0], couple[1])] = spearmanr(df[couple[0]], df[couple[1]])[0]
    return coef

#df = pd.read_csv("/home/ad/Downloads/LAB/DataMining_Lab/crop_data_anva/crop_data.csv")
#print(df.columns[2], ' ', df.columns[3])

