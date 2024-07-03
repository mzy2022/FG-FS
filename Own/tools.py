import numpy as np
import pandas as pd


def insert_generated_feature_to_original_feas(feas, f, f_name=None):
    """
    将新生成的feature插入到原来的dataframe中
    :param feas: 原dataframe
    :param f: 新feature
    :param f_name: 新feature的name，有代表feature为ndarray
    :return: 新生成的dataframe
    """
    if f_name:
        f = pd.DataFrame(f, columns=[f_name])
    y_label = pd.DataFrame(feas[feas.columns[len(feas.columns) - 1]])
    y_label.columns = [feas.columns[len(feas.columns) - 1]]
    feas = feas.drop(columns=feas.columns[len(feas.columns) - 1])
    final_data = pd.concat([feas, f, y_label], axis=1)
    return final_data

def feature_state_generation_des(X):
    """
    :param X:dataframe
    :return:
    """
    feature_matrix = []
    for i in range(8):
        feature_matrix += list(X.astype(np.float64).describe().iloc[i, :].describe().fillna(0).values)
    return feature_matrix


# X = pd.DataFrame({'a':[1,2,3],'B':[4,5,6]})
# print(len(feature_state_generation_des(X)))
