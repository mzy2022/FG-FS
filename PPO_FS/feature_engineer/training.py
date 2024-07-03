import random

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, mutual_info_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold


def sample(args,train_data_x, train_data_y, feature_nums,adj_matrix,ppo, worker, device):
    init_x_state = torch.from_numpy(train_data_x.values).float().transpose(0, 1).to(device)
    init_y_state = torch.from_numpy(train_data_y.values).float().transpose(0, 1).to(device)
    actions, log_prob, action_softmax = ppo.choose_action(init_x_state, init_y_state,feature_nums,adj_matrix)

    feature_nums = train_data_x.shape[1]
    selected_numbers = actions.cpu().numpy()
    adj_matrix = np.eye(feature_nums)
    for num in selected_numbers:
        adj_matrix[:, num] = 1

    worker.states_x = init_x_state
    worker.states_y = init_y_state
    worker.actions = actions
    worker.log_prob = log_prob
    worker.action_softmax = action_softmax
    worker.adj_matrix = adj_matrix
    return worker


def get_reward(args, worker, X_train, Y_train, X_val, Y_val):
    #TODO:需要更新一下

    actions = np.array(worker.actions.cpu())
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X_train.iloc[:, actions], Y_train)
    y_pred = model.predict(X_val.iloc[:, actions])
    Y_val = Y_val.values
    accuracy = accuracy_score(y_pred, Y_val)
    worker.reward = accuracy
    return worker,actions,accuracy

    # actions = np.array(worker.actions.cpu())
    # X = pd.concat([X_train, X_val], axis=0)
    # X = X.iloc[:,actions]
    # y = pd.concat([Y_train, Y_val], axis=0)
    # actions = np.array(worker.actions.cpu())
    # clf = RandomForestClassifier(n_estimators=10, random_state=0)
    # scores = cross_val_score(clf, X, y, scoring='f1_micro', cv=5)
    # scores = np.mean(scores)
    # worker.reward = scores
    # return worker, actions, scores


# def node_coverage_reward(selected_nodes, original_graph):
#     covered_nodes = set(selected_nodes)
#     for node in selected_nodes:
#         covered_nodes.update(original_graph.neighbors(node))
#     coverage_rate = len(covered_nodes) / original_graph.number_of_nodes()
#     return coverage_rate
#
#
# # 边覆盖率奖励：
# def edge_coverage_reward(selected_nodes, original_graph):
#     covered_edges = 0
#     total_edges = original_graph.number_of_edges()
#     for node in selected_nodes:
#         for neighbor in original_graph.neighbors(node):
#             if neighbor in selected_nodes:
#                 covered_edges += 1
#     coverage_rate = covered_edges / total_edges
#     return coverage_rate
#
# # 保持度奖励（如聚类系数）：
# import networkx as nx
#
# def clustering_coefficient_reward(selected_nodes, original_graph):
#     subgraph = original_graph.subgraph(selected_nodes)
#     original_clustering = nx.average_clustering(original_graph)
#     subgraph_clustering = nx.average_clustering(subgraph)
#     reward = 1 - abs(original_clustering - subgraph_clustering)
#     return reward
#
#
# # 平均路径长度奖励：
# def average_path_length_reward(selected_nodes, original_graph):
#     subgraph = original_graph.subgraph(selected_nodes)
#     try:
#         original_apl = nx.average_shortest_path_length(original_graph)
#         subgraph_apl = nx.average_shortest_path_length(subgraph)
#         reward = 1 - abs(original_apl - subgraph_apl) / original_apl
#     except nx.NetworkXError:
#         reward = 0  # 如果子图不是连通的，奖励为0
#     return reward