from functools import reduce
import sklearn
from .library import *
from scipy.stats import chi2_contingency, chi2

def chi_score_2Features(contigency_table, significance = 0.05):
    stat, p, df, expected = chi2_contingency(contigency_table)
    return p

def chi_score_2Columns(df, col1, col2, significance = 0.05):
    if not is_categorical_columns((df[df.columns[col1]])) or not is_categorical_columns(df[df.columns[col2]]):
        raise Exception(f'Value of {df.columns[col1]} or {df.columns[col2]} have to be categorical')
    else:
        contigency_table = create_contigency_from_2CA(df, col1, col2)
    p_value = chi_score_2Features(contigency_table, significance)
    if (p_value >= significance):
        print(f"Fail to reject H0 that {df.columns[col1]} and {df.columns[col2]} are independent")
    else:
        print(f"Reject H0 that {df.columns[col1]} and {df.columns[col2]} are independent")
    return p_value

#calculate p-value of 2 categorical variable
def chi_score(df, is_contigence_table = True, significance = 0.05, col1= -1, col2 = -1, arr = None):
    # RAISE EXCEPTION IF INPUT DATA IS NEITHER DATAFRAME NOR BUNCH
    if not isinstance(df, (pd.core.frame.DataFrame, sklearn.utils.Bunch)):
        raise Exception('Input data needs to be DATAFRAME or BUNCH')

    # CONVERT BUNCH TO DATAFRAME
    if isinstance(df, sklearn.utils.Bunch):
        df = convert_Bench2Dataframe(df)

    # CALCULATE SPECIFIC COLUMNS
    if not is_contigence_table:
        if col1 != -1 and col2 != -1:
            return chi_score_2Columns(df, col1, col2, significance)
        elif arr == None:
            raise Exception('Enter 2 column numbers or Array of columns you want to calculate Chi-score')
        else:
            dic = dict()
            for couple in combination(arr):
                couple = np.array(couple, dtype= np.int)
                dic[(df.columns[couple[0]], df.columns[couple[1]])] = chi_score_2Columns(df, couple[0], couple[1], significance)
            return dic
    else:
        p_value = chi_score_2Features(df)
        return p_value

#Example:
#df = pd.read_csv("/home/ad/Downloads/LAB/DataMining_Lab/crop_data_anva/crop_data.csv")
#print(chi_score(df, is_contigence_table=False, arr = [0,1,2]))