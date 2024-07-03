import numpy as np
from scipy.special import expit
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

O1 = ['sqrt', 'square', 'sin', 'cos', 'tanh', 'abs', 'power3', 'round', 'sigmoid', 'log', 'reciprocal', 'none']
O2 = ['+', '-', '*', '/', 'none']
O3 = ['combine', 'none']


def divide(col1, col2):
    '''
        :type col1,col2: list or np.array
        :rtype: np.array,shape = (len(array),2)
        '''
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        col_d1 = np.array([col1[idx] / col2[idx] if col2[idx] != 0 else 0 for idx in range(len(col1))])
        col_d1 = replace_abnormal(col_d1)
        return col_d1
    except:
        raise ValueError('Value type error,check feature type')


def replace_abnormal(col):
    """"""
    #
    # mean,std = np.mean(col),np.std(col)
    # floor,upper = mean - 3 * std, mean + 3 * std
    # col_replaced = [float(np.where(((x<floor)|(x>upper)), mean, x)) for x in col]
    #
    percent_25, percent_50, percent_75 = np.percentile(col, (25, 50, 75))
    IQR = percent_75 - percent_25
    floor, upper = percent_25 - 1.5 * IQR, percent_75 + 1.5 * IQR
    col_replaced = [float(np.where((x < floor), floor, x)) for x in col]
    col_replaced = [float(np.where((x > upper), upper, x)) for x in col_replaced]
    return np.array(col_replaced)


def cube(x):
    return x ** 3


def none(x,y):
    return x


def justify_operation_type(op):
    if op == 'sqrt':
        op = np.sqrt
    elif op == 'none':
        op = none
    elif op == 'abs':
        op = np.abs
    elif op == 'power3':
        op = cube
    elif op == 'round':
        op = np.round
    elif op == 'reciprocal':
        op = np.reciprocal
    elif op == 'square':
        op = np.square
    elif op == 'sin':
        op = np.sin
    elif op == 'cos':
        op = np.cos
    elif op == 'tanh':
        op = np.tanh
    elif op == '+':
        op = np.add
    elif op == '-':
        op = np.subtract
    elif op == '/':
        op = divide
    elif op == '*':
        op = np.multiply
    elif op == 'exp':
        op = np.exp
    elif op == 'sigmoid':
        op = expit
    elif op == 'log':
        op = np.log
    else:
        print('Please check your operation!')
    return op


def combine(col1, col2, fe_names):
    fe_names = list(fe_names)
    col1 = col1[:, np.newaxis]  # 将col1转换为二维数组
    col2 = col2[:, np.newaxis]  # 将col2转换为二维数组
    ori_fes = np.concatenate([col1, col2], axis=1)
    cb_df = pd.DataFrame(ori_fes, columns=fe_names, dtype='int').astype(str)
    uniuqe_idx = cb_df.groupby(fe_names).count().reset_index()[fe_names]
    uniuqe_idx = uniuqe_idx.sort_values(by=fe_names, ascending=True).astype(str)
    uniuqe_idx['keys'] = uniuqe_idx[fe_names].apply(lambda x: ''.join(x), axis=1)
    uniuqe_idx['coding'] = range(1, len(uniuqe_idx) + 1)
    col_unique_dict = dict(zip(uniuqe_idx['keys'], uniuqe_idx['coding']))
    cb_df = pd.merge(cb_df, uniuqe_idx[fe_names + ['coding']], on=fe_names, how='left')
    combine_col = cb_df['coding'].values.astype(int)
    return combine_col
