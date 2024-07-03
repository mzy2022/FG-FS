from collections import Counter

import numpy as np


def categories_to_int(col: np.array, col_index: int = None):
    col = np.array(col).reshape(-1)
    unique_type = np.array(sort_count(list(col)))
    categories_map = {}
    for i, type in enumerate(unique_type):
        categories_map[type] = i
    categories_map = dict(sorted(categories_map.items(), key=lambda x: x[1]))
    new_fe = np.array([categories_map[x] for x in col])
    return new_fe


def sort_count(vars: list) -> list:
    count = dict(Counter(vars))
    sorted_count = dict(sorted(count.items(), key=lambda x: x[1], reverse=True))
    sorted_keys = [key for key in sorted_count.keys()]
    return sorted_keys


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
            if x > fre_list[i] and x <= fre_list[i + 1]:
                return i + 1


def remove_duplication(data):
    """
    Remove duplicated columns

    :param data: pd.DataFrame
    :return: pd.DataFrame or np.array, sorted index of duplicated columns
    """
    _, idx = np.unique(data, axis=1, return_index=True)
    y = data.iloc[:, np.sort(idx)]
    return y, np.sort(idx)
