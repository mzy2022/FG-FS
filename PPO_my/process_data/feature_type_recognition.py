import pandas as pd
FEATURE_TYPE = {}
FEATURE_TYPE['cat'] = 'cat'
FEATURE_TYPE['num'] = 'num'

class Feature_type_recognition:
    def __init__(self):
        self.df = None
        self.feature_type = None

    def fit(self, df):
        self.df = df
        self.feature_type = {}
        for col in self.df.columns:
            cur_type = get_data_type(self.df, col)
            self.feature_type[col] = cur_type
        return self.feature_type

def get_data_type(df,col):
    if df[col].dtypes == object or df[col].dtypes == bool or str(df[col].dtypes) == 'category':
        if not df[col].fillna(df[col].mode()).apply(lambda x: len(str(x))).astype('float').mean() > 25:
            return FEATURE_TYPE['cat']
    if 'int' in str(df[col].dtype) or 'float' in str(df[col].dtype):
        return FEATURE_TYPE['num']