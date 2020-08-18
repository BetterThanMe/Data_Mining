from .library import *
from .Pearson_coeffcient import *
from .Spearman_coefficient import *
from .Kendall_coefficient import *
from .Mutual_information import *
from .Chi_Squared_Test import *
from .ANOVA_Test import *

class Filter:
    def __init__(self, df):
        self.df = df

    def _pearson(self, col1 = -1, col2 = -1):
        return pearson(self.df, col1, col2)

    def _spearman(self, col1 = -1, col2 = -1, ranked = False):
        return spearman(self.df, col1, col2, ranked)

    def _kendall(self, col1 = -1, col2 = -1, ranked = False):
        return kendall(self.df, col1, col2, ranked)

    def _mutualInfo(self, is_categiries = True, col = -1, target = None):
        return mutual_info(self.df, is_categiries, col, target)

    def _chi(self, is_contigence_table = True, significance = 0.05, col1  = -1, col2 = -1, arr = None):
        return chi_score(self.df, is_contigence_table, significance, col1, col2, arr)

    def _anova(self, is_1Feature = False, features = None, targets = None):
        return anova(self.df, is_1Feature, features, targets)

    def __getitem__(self):
        return self.df

    def __len__(self):
        return len(df)

#df = pd.read_csv("/home/ad/Downloads/LAB/DataMining_Lab/crop_data_anva/crop_data.csv")
#filter = Filter(df)

#print(filter._anova(is_1Feature= False, features=[0,1,2], targets= 3))