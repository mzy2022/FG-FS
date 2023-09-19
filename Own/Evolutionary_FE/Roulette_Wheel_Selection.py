import random
import sys

import numpy as np

sys.path.append("..")
import pandas as pd

from feature_cluster import cluster_features, cluster_features_1
from DNA_Fitness import fitness_score


"""
通过轮盘赌算法，根据适应度函数，对生成的聚类进行轮盘赌选择，选择k个留下的聚类
"""


def wheel_selection(ori_df, y, task_type):
    cluster_dict = cluster_features(ori_df)
    new_cluster_dict = dict()
    f_score_dict = dict()
    for k, v in cluster_dict.items():
        new_data = ori_df.iloc[:, v]
        score = fitness_score(new_data,y, task_type)
        f_score_dict[k] = score
    n = len(cluster_dict)


    if n < 3:
        return cluster_dict
    elif n > 10:
        new_cluster_list = generate_new_cluster(f_score_dict,n-2)
    else:
        new_cluster_list = generate_new_cluster(f_score_dict, n - 1)

    for k,v in cluster_dict.items():
        if k in new_cluster_list:
            new_cluster_dict[k] = v
    return new_cluster_dict




def generate_new_cluster(f_score_dict,n):
    new_cluster_list = []
    sorted_f_score_list = sorted(f_score_dict.items(), key=lambda x: x[1])
    while len(new_cluster_list) < n:
        res_f_list = sorted_f_score_list.copy()
        sum_of_value = 0
        score_dict = dict()
        p_dict = dict()
        for tup in res_f_list:
            sum_of_value += tup[1]
        for k,v in sorted_f_score_list:
            score_dict[k] = v/sum_of_value
            p_dict[k] = sum(score_dict.values())
        my_list = [random.random() for _ in range(len(res_f_list))]
        score_dict_list = res_f_list.copy()
        p_list = list(p_dict.items())
        for num,i in enumerate(my_list):
            if i < p_list[num][1] and len(new_cluster_list) < n:
                new_cluster_list.append(res_f_list[num][0])
                score_dict_list.remove(res_f_list[num])
            elif len(new_cluster_list) >= n:
                break
        sorted_f_score_list = score_dict_list.copy()
    return new_cluster_list

df = pd.read_csv('hepatitis.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
m = wheel_selection(X,y,'cls')
print(m)
