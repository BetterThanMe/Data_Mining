from library import *
from scipy.stats import f_oneway
#df = pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/difficile.csv")

df = pd.read_csv("/home/ad/Downloads/DataMining_Lab/Filter_Methods/final.csv")
#CALCULATE ANOVA SCORE WITH A FEATURES CORRESPONDING WITH CONTINUOUS TARGET VARIABLE
def anova(df):
    dic = dict()
    if (isinstance(df, pd.DataFrame)):
        for attri in df.columns:
            dic[attri] = np.array(df[attri])
    elif (isinstance(df, dict)):
        dic = df
    else:
        raise Exception("Input data is not in the right form")

    str_command = 'stat, p = f_oneway('
    for i in dic:
        str_command += 'dic['+str(i)+']'
        str_command += '' if (i==len(df)-1) else ','
    str_command += ')'
    exec (str_command, globals())
    return stat, p

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

dic = create_anova_dic(df['Holiday'], df['Increase'])
print(anova(dic))
