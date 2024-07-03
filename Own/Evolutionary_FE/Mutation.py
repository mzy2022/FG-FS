import numpy as np
import pandas as pd
from Own.feature_eng.feature_computation import O1,justify_operation_type
from Own.feature_eng.feature_cluster import cluster_features,cluster_features_1
from Own.feature_eng.feature_selection import feature_selection_list
"""
特征变异操作，对每一个特征聚类进行变异，返回新的特征集
"""

def mutation(cluster_dict, process_data, target,op_lists,data_name,task_type):
    """

    :param f_cluster: dataframe
    :param label: 标签，dataframe
    :param op: 操作符
    :param f_cluster_name: dataframe的列名
    :param label_name: 标签名称
    :param task_type: 任务类型
    :return:
    """

    cluster_list = list(cluster_dict.items())
    f_new, f_new_name = [], []
    new_cluster = dict()
    num_i = 0
    des_name = []
    new_feature_dict = dict()
    new_cluster_dict = dict()
    for num_op_list, op_list in enumerate(op_lists.values()):
        f_new, f_new_name = [], []
        feas, feas_names = [], []
        temporary_cluster_dict = dict()
        for num,op in enumerate(op_list):
            temporary_cal_list = []
            temporary_cal_name_list = []
            op_sign = justify_operation_type(op)
            v = cluster_list[num][1]
            cluster_data = process_data.iloc[:,v].values             #cluster_data 为ndarray
            if op == 'sqrt':
                for i in range(cluster_data.shape[1]):
                    if np.sum(cluster_data[:, i] < 0) == 0:
                        temporary_cal_list.append(op_sign(cluster_data[:, i]))
                        temporary_cal_name_list.append(str(data_name[v[i]]) + '_' + str(op))
                        des_name.append(str(data_name[v[i]]) + '_' + str(op))
                # new_cluster[num] = des_name
                # num_i += 1
                # des_name = []

            elif op == 'reciprocal':
                for i in range(cluster_data.shape[1]):
                    if np.sum(cluster_data[:, i] == 0) == 0:
                        temporary_cal_list.append(op_sign(cluster_data[:, i]))
                        temporary_cal_name_list.append(str(data_name[v[i]]) + '_' + str(op))
                        des_name.append(str(data_name[v[i]]) + '_' + str(op))
                # new_cluster[num] = des_name
                # num_i += 1
                # des_name = []

            elif op == 'log':
                for i in range(cluster_data.shape[1]):
                    if np.sum(cluster_data[:, i] <= 0) == 0:
                        temporary_cal_list.append(op_sign(cluster_data[:, i]))
                        temporary_cal_name_list.append(str(data_name[v[i]]) + '_' + str(op))
                        des_name.append(str(data_name[v[i]]) + '_' + str(op))
                # new_cluster[num] = des_name
                # num_i += 1
                # des_name = []

            elif op == 'none':
                for i in range(cluster_data.shape[1]):
                    temporary_cal_list.append(cluster_data[:, i])
                    temporary_cal_name_list.append(str(data_name[v[i]]))
                    des_name.append(str(data_name[v[i]]))
                # new_cluster[num] = des_name
                # num_i += 1
                # des_name = []
            else:
                for i in range(cluster_data.shape[1]):
                    temporary_cal_list.append(op_sign(cluster_data[:,i]))
                    temporary_cal_name_list.append(str(data_name[v[i]]) + '_' + str(op))
                    des_name.append(str(data_name[v[i]]) + '_' + str(op))
                # new_cluster[num] = des_name
                # num_i += 1
                # des_name = []

            if len(temporary_cal_list) > 50:
                temporary_cal_list = np.array(temporary_cal_list)
                temporary_cal_list = temporary_cal_list.T
                fes_new, fes_names_new = feature_selection_list(temporary_cal_list, target, temporary_cal_name_list,
                                                                task_type=task_type)
                feas_names.append(fes_names_new)
                feas.append(fes_new)
                temporary_cluster_dict[num] = fes_names_new
                num_i += 1
                des_name = []

            else:
                feas_names.append(temporary_cal_name_list)
                feas.append(temporary_cal_list)
                temporary_cluster_dict[num] = des_name
                num_i += 1
                des_name = []

        new_cluster[num_op_list] = temporary_cluster_dict
        final_list = [j for i in feas for j in i]
        final_name = [j for i in feas_names for j in i]
        feas = np.array(final_list)
        feas_names = np.array(final_name)
        if len(feas) == 0 and len(feas_names) == 0:
            new_df = process_data
            new_feature_dict[num_op_list] = new_df
            new_cluster[num_op_list] = cluster_dict

        else:
            new_df = pd.DataFrame(feas.T, columns=feas_names)
            new_feature_dict[num_op_list] = new_df
            new_cluster[num_op_list] = {k: [list(new_df.columns).index(e) for e in v] for k, v in
                                        new_cluster[num_op_list].items()}
        # feas = np.array(f_new)
        # feas_names = np.array(f_new_name)
        # new_df = pd.DataFrame(feas.T, columns=feas_names)
        # new_cluster = {k: [list(new_df.columns).index(e) for e in v] for k, v in new_cluster.items()}
        # new_cluster = cluster_features(new_df,target)
        # new_feature_dict[num_op_list] = new_df
        # new_cluster_dict[num_op_list] = new_cluster

    return new_feature_dict,new_cluster



# data = pd.read_csv("hepatitis.csv")
# X = data.iloc[:,:-1]
# y = data.iloc[:,-1]
# X_name = X.columns
# dis = cluster_features(X)
#
# df = mutation(dis,X,y,['power3','sqrt','square', 'sin', 'cos', 'tanh', 'none'],X_name,'cls')
# print(df)