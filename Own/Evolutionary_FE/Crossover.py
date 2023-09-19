import numpy as np
import pandas as pd

from DNA_Fitness import fitness_score
from Own.Evolutionary_FE.Roulette_Wheel_Selection import wheel_selection
from feature_cluster import cluster_features
from Own.feature_eng.feature_computation import O2,justify_operation_type

"""
用于对每个聚类进行二元操作，按概率选择其他的聚类进行交叉
"""


def crossover(f_cluster,label,op,f_cluster_name,task_type):
    mutation_dict = wheel_selection(f_cluster, label, task_type)
    assert op in O1
    cluster_dict = cluster_features(X)
    f_score_dict = dict()
    ori_dict = dict()
    for k,v in cluster_dict:
        new_data = X.iloc[:,v]
        score = fitness_score(new_data,label,task_type)
        f_score_dict[k] = score


def operate_two_features(f_cluster1, f_cluster2, op, f_names1, f_names2):
    assert op in O2
    if op == '/' and np.sum(f_cluster2 == 0) > 0:
        return None, None
    op_func = justify_operation_type(op)
    feas, feas_names = [], []
    for i in range(f_cluster1.shape[1]):
        for j in range(f_cluster2.shape[1]):
            feas.append(op_func(f_cluster1[:, i], f_cluster2[:, j]))
            feas_names.append(str(f_names1[i]) + op + str(f_names2[j]))
    feas = np.array(feas)
    feas_names = np.array(feas_names)
    return feas.T, feas_names

def insert_generated_feature_to_original_feas(feas, f):
    y_label = pd.DataFrame(feas[feas.columns[len(feas.columns) - 1]])
    y_label.columns = [feas.columns[len(feas.columns) - 1]]
    feas = feas.drop(columns=feas.columns[len(feas.columns) - 1])
    final_data = pd.concat([feas, f, y_label], axis=1)
    return final_data





