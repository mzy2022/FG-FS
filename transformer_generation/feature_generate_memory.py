from __future__ import absolute_import

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from DQN_my2.process_data.utils_memory import ff


def binning_with_tree(ori_fe: np.array, label: np.array):
    boundry = []
    clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                 max_leaf_nodes=6,  # 最大叶子节点数
                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比
    fe = ori_fe.reshape(-1, 1)
    clf.fit(fe, label.astype("int"))  # 训练决策树

    n_nodes = clf.tree_.node_count  # 决策树的节点数
    children_left = clf.tree_.children_left  # node_count大小的数组，children_left[i]表示第i个节点的左子节点
    children_right = clf.tree_.children_right  # node_count大小的数组，children_right[i]表示第i个节点的右子节点
    threshold = clf.tree_.threshold  # node_count大小的数组，threshold[i]表示第i个节点划分数据集的阈值
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:
            boundry.append(threshold[i])
    boundry.sort()
    if len(boundry):
        new_fe = np.array([ff(x, boundry) for x in ori_fe])
    else:
        new_fe = ori_fe
    return new_fe


def sqrt(col):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    # 需要保证输入的特征全部是数值型特征
    # 求三次方根
    try:
        sqrt_col = np.sqrt(np.abs(col))
        return sqrt_col.reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')


def power3(col):
    col = np.array(col)
    new_col = np.power(col, 3)
    return new_col.reshape(-1, 1)


def sigmoid(col):
    col = np.array(col)
    new_col = 1 / (1 + np.exp(-col))
    return new_col.reshape(-1, 1)


def tanh(col):
    col = np.array(col)
    new_col = (np.exp(col) - np.exp(-col)) / (np.exp(col) + np.exp(-col))
    print(new_col)
    exit()
    return new_col.reshape(-1, 1)


def inverse(col, memory=None):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    try:
        col = np.array(col)
        # if np.any(col == 0):
        #     return None
        new_col = np.array([1 / x if x != 0 else x for x in col])
        return new_col.reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')


def square(col):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    col = np.array(col)
    new_col = np.square(col).reshape(-1, 1)
    return new_col.reshape(-1, 1)


def abss(col):
    col = np.array(col)
    new_col = np.abs(col)
    return new_col.reshape(-1, 1)


def log(col):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    # 底数为自然底数e
    try:
        log_col = np.array([np.log(abs(x)) if abs(x) > 0 else np.log(1) for x in col])
        return log_col.reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')


def min_max(col: list or np.ndarray, col_index: int = None):
    col = np.array(col)
    min = np.min(col, axis=0)
    max = np.max(col, axis=0)
    if min == max:
        return col
    else:
        scaled = (col - min) / (max - min)
        return scaled.reshape(-1, 1)


def normalization(col: list or np.ndarray, col_index: int = None) -> np.array:
    '''
    Parameters
    ----------
    :param col: list or np.array
    Returns
    ----------
    return:
        - col:np.array
    '''
    # 特征z-core标准化
    col = np.array(col)
    mu = np.mean(col, axis=0)
    sigma = np.std(col, axis=0)
    if sigma == 0:  # while sigma is 0,return ori_col
        return col.reshape(-1, 1)
    else:
        scaled = ((col - mu) / sigma)
    return scaled.reshape(-1, 1)


# 两个数值特征的四则运算操作
def add(col1, col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    # 数值特征加法
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        return (col1 + col2).reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')


def multiply(col1, col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    # 数值特征乘法
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        return (col1 * col2).reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')


def subtract(col1, col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),2)
    '''
    # 数值特征减法，不指定被减数的话，生成的应该是两列特征
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        return np.abs(col1 - col2).reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')


def divide(col1, col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),2)
    '''
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        # if np.any(col2 == 0):
        #     return None
        col_d1 = np.array([x1 / x2 if x2 != 0 else 1 for x1, x2 in zip(col1, col2)]).reshape(-1, 1)
        return col_d1
        # return np.concatenate((col_d1, col_d2), axis=1)
    except:
        raise ValueError('Value type error,check feature type')


# def reset_value(ori_fe, c, merged_values, k):
#     '''将原始分类变量值重置为其他'''
#     for merged_value in merged_values:
#         indexs = np.argwhere(ori_fe == merged_value).reshape(-1)
#         new_value = k + c
#         ori_fe[indexs] = new_value
def get_nunique_feature(col1, col2):
    new_fe = []
    col1 = np.array(col1).reshape(-1, 1)
    col2 = np.array(col2).reshape(-1, 1)
    feature = np.concatenate([col1, col2], axis=1)
    for each in feature:
        x = each[0] + each[1]
        new_fe.append(x)
    new_fe = np.array(new_fe)
    # new_fe = (pd.DataFrame(feature).fillna(0)).groupby(0)[1].transform('nunique').values
    # le = LabelEncoder()
    # new_fe = le.fit_transform(new_fe) + 1

    # new_fe = (pd.DataFrame(new_fe).fillna(0)).groupby([0])[0].transform('count').values

    return new_fe.reshape(-1, 1)


def generate_combine_fe(ori_fe1, ori_fe2, feasible_values: dict) -> np.array:
    '''convert combine category feature to onehot feature'''
    k = len(ori_fe1)
    new_fe = np.zeros(k)
    for i in range(k):
        combine_feature_value = str(int(ori_fe1[i])) + str(int(ori_fe2[i]))
        ind = feasible_values[combine_feature_value]
        new_fe[i] = ind
    return new_fe.reshape(-1, 1)
    # col1 = np.array(ori_fe1).reshape(-1, 1)
    # col2 = np.array(ori_fe2).reshape(-1, 1)
    # feature = np.concatenate([col1, col2], axis=1)
    # new_fe = (pd.DataFrame(feature).fillna(0)).groupby(0)[1].transform('nunique').values
    # return new_fe.reshape(-1,1)


if __name__ == '__main__':
    data = pd.read_csv("SPECTF.csv")
    X = data.iloc[:, 1]
    y = data.iloc[:, 2]
    X = np.array(X.values)
    y = np.array(y.values)
    new = divide(X, y)
    print(new)

    x = get_nunique_feature(col1=[1, 2, 3], col2=[1, 0, 2])
    print(x)
