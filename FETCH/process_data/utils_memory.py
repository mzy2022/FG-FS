import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
from .memory_no_encode import Memory

# 这段代码定义了一个用于将分类特征转换为整数类型的函数categories_to_int,用于将离散特征的类别映射为整数类型
def categories_to_int(col:np.array,col_index:int = None,memory:Memory = None,isvalid:bool = False):
    """categories features to int type.

       Input original col must be discrete feature.

       Parameters
       ----------
       :param col: np.array
       :param col_index: int
       :param memory: class Memory
       :param isvalid: bool,judge whether to get information in memory
       Returns
       ----------
       returns:
           - new_fe : np.array, 1D
       """
    col = np.array(col).reshape(-1)
    if isvalid:
        categories_map = memory.category_to_int_info[col_index]
        unique_type = list(categories_map.values())
        new_fe = np.array([categories_map[x] if x in categories_map else unique_type[-1] for x in col])
        return new_fe

    # sort mapping values with frequency,high to low
    unique_type = np.array(sort_count(list(col)))
    categories_map = {}
    for i,type in enumerate(unique_type):
        categories_map[type] = i
    categories_map = dict(sorted(categories_map.items(),key=lambda x: x[1]))
    new_fe = np.array([categories_map[x] for x in col])
    if memory is not None:
        memory.category_to_int_info[col_index] = categories_map
    return new_fe


def sort_count(vars:list) -> list:
    '''
       对列表中的元素按照出现的频次从高到低排序
       :param list: 元素为int型的列表
       :return: 元素按照出现次数排序（降序）
       '''
    count = dict(Counter(vars))
    sorted_count = dict(sorted(count.items(),key = lambda x:x[1],reverse = True))
    sorted_keys = [key for key in sorted_count.keys()]
    return sorted_keys

def calculate_chi2(col, label):
    '''
       # 计算某个特征每种属性值的卡方统计值
       :type col: list or np.array, 特征列, 注意需要保证是分类特征
       :type label: list or np.array
       '''
    col = np.array(col)
    target_total = np.sum(label)
    target_len = len(label)
    # 计算样本期望值
    expect_ratio = target_total / target_len
    feature_unique_values = np.unique(col)
    chi2_dict = {}
    for value in feature_unique_values:
        # 计算各类别对应target的期望值
        indexs = np.argwhere(col == value).reshape(-1)
        target_of_value = label[indexs]
        target_of_value_sum = np.sum(target_of_value)
        target_of_value_len = len(target_of_value)
        expected_target_sum = target_of_value_len * expect_ratio
        chi2 = (target_of_value_sum - expected_target_sum) ** 2 / expected_target_sum
        chi2_dict[value] = chi2
    chi2_dict_sorted = dict(sorted(chi2_dict.items(), key=lambda x: x[1], reverse=True))
    return chi2_dict_sorted


def ff(x, fre_list):
    '''
        # 根据数值所在区间，给特征重新赋值，区间左开右闭
        :type x: float, 单个特征的值
        :type fre_list: list of floats,分箱界限
        '''
    if x <= fre_list[0]:
        return 0
    elif x > fre_list[-1]:
        return len(fre_list)
    else:
        for i in range(len(fre_list) - 1):
            if fre_list[i] < x <= fre_list[i + 1]:
                return i + 1

def combine_feature_tuples(feature_list, combine_type):
    '''
    特征组合，生成新组合特征的tuple
    :type feature_list: list
    :type combine_type: int
    :rtype: list of tuples like[(A,B),(B,C)]
    '''
    return list(combinations(feature_list, combine_type))
