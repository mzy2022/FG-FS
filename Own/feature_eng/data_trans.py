import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import datetime
from sklearn.preprocessing import MinMaxScaler

"""
这个脚本用来对数据进行清洗，包括是否进行归一化处理等
"""
FEATURE_TYPE = {}
FEATURE_TYPE['txt'] = 'txt'
FEATURE_TYPE['num'] = 'num'
FEATURE_TYPE['cat'] = 'cat'
FEATURE_TYPE['ord'] = 'ord'
FEATURE_TYPE['datetime'] = 'datetime'
FEATURE_TYPE['timestamp'] = 'timestamp'


class Pipeline_data(object):
    def __init__(self, all_data, continuous_columns, discrete_columns, do_minmax=True):
        self.continuous_columns = continuous_columns
        self.discrete_columns = discrete_columns
        self.all_data = all_data
        self.new_data = all_data
        self.do_minmax = do_minmax

    def get_data_type(self, df, col):
        if self.detect_DATETIME(df, col):
            return FEATURE_TYPE['datetime']
        if self.detect_TIMESTAMP(df, col):
            return FEATURE_TYPE['timestamp']
        if df[col].dtypes == object or df[col].dtypes == bool or str(df[col].dtypes) == 'category':
            if df[col].fillna(df[col].mode()).apply(lambda x: len(str(x))).astype('float').mean() > 25:
                return FEATURE_TYPE['txt']
            return FEATURE_TYPE['cat']
        if 'int' in str(df[col].dtype) or 'float' in str(df[col].dtype):
            return FEATURE_TYPE['num']

    def fill_data(self, all_data):
        for x in self.continuous_columns:
            all_data[x].replace('?', np.nan, inplace=True)
            all_data[x].replace("NA", np.nan, inplace=True)
            all_data[x] = self.all_data[x].astype(float)
            mean = np.nanmean(self.all_data[x])
            all_data[x].fillna(mean, inplace=True)
        for x in self.discrete_columns:
            all_data[x].replace('?', np.nan, inplace=True)
            all_data[x].replace("NA", np.nan, inplace=True)
            all_data[x] = all_data[x].astype(float)
            mean = np.nanmean(all_data[x])
            all_data[x].fillna(mean, inplace=True)
            all_data[x] = all_data[x].astype(int)
        return all_data

    def minmaxscaler(self, all_data):
        scaler = MinMaxScaler()
        # 对数据进行归一化
        normalized_data = scaler.fit_transform(all_data)
        return normalized_data

    def categories_to_int(self, col):
        '''
                :type col: list or np.array
                :rtype: np.array,shape = (len(array),1)
        '''
        unique_type = np.unique(np.array(col))
        categories_map = {}
        for i, type in enumerate(unique_type):
            categories_map[type] = i
        new_fe = np.array([categories_map[x] for x in col])
        return new_fe

    def detect_TIMESTAMP(self, df, col):
        try:
            ts_min = int(float(df.loc[~(df[col] == '') & (df[col].notnull()), col].min()))
            ts_max = int(float(df.loc[~(df[col] == '') & (df[col].notnull()), col].max()))
            datetime_min = datetime.datetime.utcfromtimestamp(ts_min).strftime('%Y-%m-%d %H:%M:%S')
            datetime_max = datetime.datetime.utcfromtimestamp(ts_max).strftime('%Y-%m-%d %H:%M:%S')
            if datetime_min > '2000-01-01 00:00:01' and datetime_max < '2030-01-01 00:00:01' and datetime_max > datetime_min:
                return True
        except:
            return False

    def detect_DATETIME(self, df, col):
        is_DATETIME = False
        if df[col].dtypes == object or str(df[col].dtypes) == 'category':
            is_DATETIME = True
            try:
                pd.to_datetime(df[col])
            except:
                is_DATETIME = False
        return is_DATETIME

    def new_pipline_main(self):
        column_list = list(self.all_data.columns)
        for col_name in column_list:
            feature_type = self.get_data_type(self.all_data, col_name)
            if feature_type == 'txt' or feature_type == 'datetime' or feature_type == 'timestamp':
                self.new_data.drop(col_name, axis=1, inplace=True)
        for col_name in self.discrete_columns:
            new_fe = self.categories_to_int(self.all_data[col_name].values)
            self.new_data[col_name] = new_fe
        self.new_data = self.fill_data(self.new_data)
        if self.do_minmax:
            self.new_data = self.minmaxscaler(self.new_data)
        return self.new_data

    def old_pipline_main(self,df):
        if self.do_minmax:
            self.new_data = self.minmaxscaler(df)
        return self.new_data
