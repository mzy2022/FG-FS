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
FEATURE_TYPE['txt']= 'txt'
FEATURE_TYPE['num']= 'num'
FEATURE_TYPE['cat']= 'cat'
FEATURE_TYPE['ord']= 'ord'
FEATURE_TYPE['datetime']= 'datetime'
FEATURE_TYPE['timestamp']= 'timestamp'
class Pipeline_data(object):
    def __init__(self, all_data,continuous_columns, discrete_columns, do_onehot=False):
        self.continuous_columns = continuous_columns
        self.discrete_columns = discrete_columns
        self.do_onehot = do_onehot
        self.all_data = all_data

    def fill_data(self):
        for x in self.continuous_columns:
            self.all_data[x].replace('?', np.nan, inplace=True)
            self.all_data[x].replace("NA", np.nan, inplace=True)
            self.all_data[x] = self.all_data[x].astype(float)
            mean = np.nanmean(self.all_data[x])
            self.all_data[x].fillna(mean, inplace=True)
        for x in self.discrete_columns:
            self.all_data[x].replace('?', np.nan, inplace=True)
            self.all_data[x].replace("NA", np.nan, inplace=True)
            self.all_data[x] = self.all_data[x].astype(float)
            mean = np.nanmean(self.all_data[x])
            self.all_data[x].fillna(mean, inplace=True)
            self.all_data[x] = self.all_data[x].astype(int)

    def categories_to_int(self, col, name):
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

    def replace_abnormal(self,col):
        percent_25, percent_50, percent_75 = np.percentile(col, (25, 50, 75))
        IQR = percent_75 - percent_25
        floor, upper = percent_25 - 1.5 * IQR, percent_75 + 1.5 * IQR
        col_replaced = [float(np.where((x < floor), floor, x)) for x in col]
        col_replaced = [float(np.where((x > upper), upper, x)) for x in col_replaced]
        return np.array(col_replaced).reshape(len(col_replaced), 1)

    def normalization(self, col, col_op):
        '''
                :type col: list or np.array
                :rtype: np.array,shape = (len(array),1)
        '''
        col = np.array(col).reshape(-1)
        mean = np.mean(col)
        std = np.std(col)
        if std != 0:
            return (col - mean) / std
        else:
            return col

    def max_min(self, col, col_op):
        '''
                :type col: list or np.array
                :rtype: np.array,shape = (len(array),1)
        '''
        col = np.array(col).reshape(-1)
        max = np.max(col)
        min = np.min(col)
        if (max - min) != 0:
            return (col - min) / (max - min)
        else:
            return col

    def detect_TIMESTAMP(self,df, col):
        try:
            ts_min = int(float(df.loc[~(df[col] == '') & (df[col].notnull()), col].min()))
            ts_max = int(float(df.loc[~(df[col] == '') & (df[col].notnull()), col].max()))
            datetime_min = datetime.datetime.utcfromtimestamp(ts_min).strftime('%Y-%m-%d %H:%M:%S')
            datetime_max = datetime.datetime.utcfromtimestamp(ts_max).strftime('%Y-%m-%d %H:%M:%S')
            if datetime_min > '2000-01-01 00:00:01' and datetime_max < '2030-01-01 00:00:01' and datetime_max > datetime_min:
                return True
        except:
            return False

    def detect_DATETIME(self,df, col):
        is_DATETIME = False
        if df[col].dtypes == object or str(df[col].dtypes) == 'category':
            is_DATETIME = True
            try:
                pd.to_datetime(df[col])
            except:
                is_DATETIME = False
        return is_DATETIME


    def binning(self, ori_fe, bins, fe_name=None, method='frequency'):
        '''

                :type ori_fe: list or np.array
                :type bins: int
                :type fe_name: str,feature_eng_bins_dict中
                :type method: str
                :rtype:1. np.array,shape = (len(array),2)，
                       2.fre_list,list of floats,
                       3.new_fe_encode,np.mat,
                '''
        ori_fe = np.array(ori_fe)
        if method == 'frequency':
            fre_list = [np.percentile(ori_fe, 100 / bins * i) for i in range(1, bins)]
            fre_list = sorted(list(set(fre_list)))
            new_fe = np.array([self.ff(x, fre_list) for x in ori_fe])
            return new_fe.reshape(-1)

        elif method == 'distance':
            umax = np.percentile(ori_fe, 99.99)
            umin = np.percentile(ori_fe, 0.01)
            step = (umax - umin) / bins
            fre_list = [umin + i * step for i in range(bins)]
            new_fe = np.array([self.ff(x, fre_list) for x in ori_fe])
            return new_fe.reshape(-1)

    def ff(self,x, fre_list):
        '''
            #
            :type x: float,
            :type fre_list: list of floats,
            '''
        if x <= fre_list[0]:
            return 0
        elif x > fre_list[-1]:
            return len(fre_list)
        else:
            for i in range(len(fre_list) - 1):
                if fre_list[i] < x <= fre_list[i + 1]:
                    return i + 1

    def recursion_freq_bins(self,count_dict,bins,total_rate,res):
        if bins > 0 and len(count_dict) > 0:
            rate = total_rate / bins
            if count_dict[0][-1] >= rate:
                res.append(count_dict[0])
                bins = bins - 1
                total_rate = total_rate - count_dict[0][-1]
                count_dict.remove(count_dict[0])
                res = self.recursion_freq_bins(count_dict,bins,total_rate,res)

            else:
                if len(count_dict) > 1:
                    count_dict[0] = (*count_dict[0][:-1] + count_dict[1][:-1], count_dict[0][-1] + count_dict[1][-1])
                    count_dict.remove(count_dict[1])
                    res = self.recursion_freq_bins(count_dict,bins,total_rate,res)
                else:
                    res.append(count_dict[0])
        return res


    def discrete_freq_bins(self,ori_fe,bins,fe_name):
        ori_fe = ori_fe.reshape(-1).astype(int)
        count_dict = dict(Counter(ori_fe))
        t_num = len(ori_fe)
        count_dict = {k: v / t_num for k,v in count_dict.items()}
        count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

        total_rate = 1
        res = []
        res = self.recursion_freq_bins(count_dict, bins, total_rate, res)
        merge_dict = {idx: list(tp[:-1]) for idx, tp in enumerate(res)}
        map_dict = {v: k for k, v in merge_dict.items() for v in v}

        bins_fe = pd.Series(ori_fe).map(map_dict).values
        return bins_fe.reshape(-1)

    def check_is_continuous(self,ori_fes,fe_names,continuous_columns,continuous_bins):
        for idx,name in enumerate(fe_names):
            if name in continuous_columns:
                bins = continuous_bins[name]
                if len(np.unique(ori_fes[:,idx])) > bins:
                    fes_bins ,_ = self.binning(ori_fes[:,idx],bins,fe_name = name,method = 'frequency')
                    ori_fes[:,idx] = fes_bins.reshape(len(fes_bins))
        return ori_fes

    def get_data_type(self,df, col):
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

    def onehot_encoder(self, ori_fe, col_name):
        ori_fe = np.array(ori_fe).reshape(-1, 1)
        encoder = OneHotEncoder(handle_unknown='ignore')
        enc = encoder.fit(ori_fe)
        onehot_fe = enc.transform(ori_fe).toarray()
        return onehot_fe

    def minmaxscaler(self):
        scaler = MinMaxScaler()
        # 对数据进行归一化
        normalized_data = scaler.fit_transform(self.all_data)
        self.processed_data = normalized_data


    def pipline_main(self):
        self.fill_data()

        self.minmaxscaler()































































