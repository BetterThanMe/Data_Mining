from scipy.stats import pearsonr
from .library import *
import sklearn

#CALCULATE PEARSON COEFFICIENT OF DATAFRAME FUNCTION FOR ALL COMBINATION 2 FEATURES OF DATAFRAME
def pearson(df, col1 = -1, col2 = -1):
    #RAISE EXCEPTION IF INPUT DATA IS NEITHER DATAFRAME NOR BUNCH
    if not isinstance(df, (pd.core.frame.DataFrame, sklearn.utils.Bunch)):
        raise Exception('Input data needs to be DATAFRAME or BUNCH')

    # CONVERT BUNCH TO DATAFRAME
    if isinstance(df, sklearn.utils.Bunch):
        df = convert_Bench2Dataframe(df)

    # CALCULATE SPECIFIC COLUMNS
    if (col1 != -1 and col2 != -1):
        print(f'Pearson coefficient between: {df.columns[col1]} and {df.columns[col2]}')
        return pearsonr(df[df.columns[col1]], df[df.columns[col2]])[0]

    coef = dict()
    for couple in combination(df.columns):
        coef[(couple[0], couple[1])] = pearsonr(df[couple[0]], df[couple[1]])[0]
    return coef

#READ DATA TABLE FROM CSV FILE
#df = pd.read_csv("/home/ad/Downloads/LAB/DataMining_Lab/crop_data_anva/crop_data.csv")


