from collections import defaultdict
from math import sqrt

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_selection import mutual_info_regression

'''
这个脚本用来聚类
'''


def cluster_features(X,y):
    return_dict = dict()
    n_cluster = int(sqrt(X.shape[1]))
    kmeans = KMeans(n_clusters=n_cluster, n_init=10)
    # 对数据进行聚类
    kmeans.fit(X.T)
    # 获取聚类结果
    labels = kmeans.labels_
    for num, i in enumerate(labels):
        return_dict.setdefault(i, []).append(num)
    return return_dict


def cluster_features_1(features, y, cluster_num=2, mode='k'):
    if mode == 'c':
        return _wocluster_features(features, y, cluster_num)
    else:
        return _cluster_features(features, y, cluster_num)


# 用于将每个特征列作为一个独立的聚类，并返回聚类结果
def _wocluster_features(features, y, cluster_num=2):
    clusters = defaultdict(list)
    for ind, item in enumerate(range(features.shape[1])):
        clusters[item].append(ind)
    return clusters


# 用于将特征进行聚类，并返回聚类结果
def _cluster_features(features, y, cluster_num=2):
    features = np.array(features)
    y = np.array(y)
    k = int(np.sqrt(features.shape[1]))
    if k > 1:
        features = feature_distance(features, y)
        features = features.reshape(features.shape[0], -1)
        clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='single').fit(features)
        labels = clustering.labels_
        clusters = defaultdict(list)
        for ind, item in enumerate(labels):
            clusters[item].append(ind)
    else:
        return {0:[0]}
    return clusters


def feature_distance(feature, y):
    return mi_feature_distance(feature, y)


def mi_feature_distance(features, y):
    dis_mat = []
    for i in range(features.shape[1]):
        tmp = []
        for j in range(features.shape[1]):
            tmp.append(np.abs(mutual_info_regression(features[:, i].reshape(-1, 1), y) - mutual_info_regression(
                features[:, j].reshape(-1, 1), y))[0] / (
                               mutual_info_regression(features[:, i].reshape(-1, 1), features[:, j].reshape(-1, 1))[
                                   0] + 1e-05))
        dis_mat.append(np.array(tmp))
    dis_mat = np.array(dis_mat)
    return dis_mat
