import numpy as np
import pandas as pd

from Own.Evolutionary_FE.DNA_Fitness import fitness_score
from Own.feature_eng.feature_cluster import cluster_features,cluster_features_1
from Own.feature_eng.feature_computation import O2,justify_operation_type,O3,combine
from sklearn.feature_selection import mutual_info_classif
from Own.feature_eng.feature_selection import feature_selection_list
from Own.feature_eng.feature_cluster import cluster_features
from Own.feature_eng.feature_computation import O1,justify_operation_type
from Own.feature_eng.feature_cluster import cluster_features,cluster_features_1
from Own.feature_eng.feature_selection import feature_selection_list


def update_binary(cluster_dict,ori_df,op_list,label,task_type):
    """

    :param cluster_dict:
    :param ori_df:
    :param label:
    :param op_list:
    :param ori_df_name:
    :param task_type:
    :return: 新生成的dataframe，不包括label
    """
    ori_df_name = ori_df.columns
    f_score_dict = cal_fitness_score(ori_df,label,task_type,cluster_dict)
    cluster_list = list(cluster_dict.items())
    new_cluster = dict()
    des_name = []
    num_i = 0
    feas, feas_names = [], []
    for num,op in enumerate(op_list):
        temporary_cal_list = []
        temporary_cal_name_list = []
        if op in O2:
            v = cluster_list[num][1]
            cluster_data = ori_df.iloc[:, v].values
            cluster_data_name = ori_df_name[v]
            choise_num_list = choise_candidate_cluster(f_score_dict,cluster_dict)
            candidate_cluster_data = ori_df.iloc[:,choise_num_list].values
            candidate_cluster_data_name = ori_df_name[choise_num_list]
            if op == '/' and np.sum(candidate_cluster_data == 0) > 0:
                for i in range(cluster_data.shape[1]):
                    temporary_cal_list.append(cluster_data[:, i])
                    des_name.append(str(cluster_data_name[i]))
                    temporary_cal_name_list.append(str(cluster_data_name[i]))
            else:
                op_func = justify_operation_type(op)
                for i in range(cluster_data.shape[1]):
                    for j in range(candidate_cluster_data.shape[1]):
                        des_name.append(str(cluster_data_name[i]) + op + str(candidate_cluster_data_name[j]))
                        temporary_cal_list.append(op_func(cluster_data[:, i], candidate_cluster_data[:, j]))
                        temporary_cal_name_list.append(str(cluster_data_name[i]) + op + str(candidate_cluster_data_name[j]))

            # if len(temporary_cal_list) > 50:
            #     temporary_cal_list = np.array(temporary_cal_list)
            #     temporary_cal_list = temporary_cal_list.T
            #     fes_new, fes_names_new = feature_selection_list(temporary_cal_list,label,temporary_cal_name_list,task_type=task_type)
            #     feas_names.append(fes_names_new)
            #     feas.append(fes_new)
            #     temporary_cluster_dict[num] = fes_names_new
            #     num_i += 1
            #     des_name = []
            #
            # else:
            feas_names.append(temporary_cal_name_list)
            feas.append(temporary_cal_list)


        elif op in O3:
            op_sign = justify_operation_type(op)
            feas, feas_names = [], []
            for k, v in cluster_dict.items():
                cluster_data = ori_df.iloc[:, v].values
                cluster_data_name = ori_df_name[v]
                choise_num_list = choise_candidate_cluster(f_score_dict, cluster_dict)
                candidate_cluster_data = ori_df.iloc[:, choise_num_list].values
                candidate_cluster_data_name = ori_df_name[choise_num_list]
                if op == 'combine':
                    for i in range(cluster_data.shape[1]):
                        for j in range(candidate_cluster_data.shape[1]):
                            x = op_sign(cluster_data[:, i], candidate_cluster_data[:, j])
                            feas.append(x)
                            feas_names.append(str(cluster_data_name[i]) + op + str(candidate_cluster_data_name[j]))
                feas = np.array(feas)
            feas_names = np.array(feas_names)
            new_cluster[num] = feas_names
            num_i += 1

    final_list = [j for i in feas for j in i]
    final_name = [j for i in feas_names for j in i]
    feas = np.array(final_list)
    feas_names = np.array(final_name)
    new_df = pd.DataFrame(feas.T, columns=feas_names)
    return new_df



def update_unary(cluster_dict,ori_df,op_list,label,task_type):
    """

    :param f_cluster: dataframe
    :param label: 标签，dataframe
    :param op: 操作符
    :param f_cluster_name: dataframe的列名
    :param label_name: 标签名称
    :param task_type: 任务类型
    :return:
    """
    data_name = ori_df.columns
    cluster_list = list(cluster_dict.items())
    num_i = 0
    des_name = []


    feas, feas_names = [], []
    temporary_cluster_dict = dict()
    for num,op in enumerate(op_list):
        temporary_cal_list = []
        temporary_cal_name_list = []
        op_sign = justify_operation_type(op)
        v = cluster_list[num][1]
        cluster_data = ori_df.iloc[:,v].values             #cluster_data 为ndarray
        if op == 'sqrt':
            for i in range(cluster_data.shape[1]):
                if np.sum(cluster_data[:, i] < 0) == 0:
                    temporary_cal_list.append(op_sign(cluster_data[:, i]))
                    temporary_cal_name_list.append(str(data_name[v[i]]) + '_' + str(op))
                    des_name.append(str(data_name[v[i]]) + '_' + str(op))


        elif op == 'reciprocal':
            for i in range(cluster_data.shape[1]):
                if np.sum(cluster_data[:, i] == 0) == 0:
                    temporary_cal_list.append(op_sign(cluster_data[:, i]))
                    temporary_cal_name_list.append(str(data_name[v[i]]) + '_' + str(op))
                    des_name.append(str(data_name[v[i]]) + '_' + str(op))


        elif op == 'log':
            for i in range(cluster_data.shape[1]):
                if np.sum(cluster_data[:, i] <= 0) == 0:
                    temporary_cal_list.append(op_sign(cluster_data[:, i]))
                    temporary_cal_name_list.append(str(data_name[v[i]]) + '_' + str(op))
                    des_name.append(str(data_name[v[i]]) + '_' + str(op))


        elif op == 'none':
            for i in range(cluster_data.shape[1]):
                temporary_cal_list.append(cluster_data[:, i])
                temporary_cal_name_list.append(str(data_name[v[i]]))
                des_name.append(str(data_name[v[i]]))

        else:
            for i in range(cluster_data.shape[1]):
                temporary_cal_list.append(op_sign(cluster_data[:,i]))
                temporary_cal_name_list.append(str(data_name[v[i]]) + '_' + str(op))
                des_name.append(str(data_name[v[i]]) + '_' + str(op))

        feas_names.append(temporary_cal_name_list)
        feas.append(temporary_cal_list)
        temporary_cluster_dict[num] = des_name
        num_i += 1
        des_name = []


    final_list = [j for i in feas for j in i]
    final_name = [j for i in feas_names for j in i]
    feas = np.array(final_list)
    feas_names = np.array(final_name)
    if len(feas)==0 and len(feas_names)==0:
        new_df = ori_df
    else:
        new_df = pd.DataFrame(feas.T, columns=feas_names)
    return new_df



def choise_candidate_cluster(fitness_score_dict,crossover_dict):
    soft_dict = softmax(fitness_score_dict)
    choise_num = sample_from_distribution(soft_dict)
    return crossover_dict[choise_num]



def cal_fitness_score(f_cluster,label,task_type,crossover_dict):
    f_score_dict = dict()
    for k, v in crossover_dict.items():
        new_data = f_cluster.iloc[:, v]
        score = fitness_score(new_data, label, task_type)
        f_score_dict[k] = score
    return f_score_dict

def softmax(fitness_score_dict):
    soft_dict = dict()
    keys = fitness_score_dict.keys()
    values = fitness_score_dict.values()
    v = list(values)
    e_x = np.exp(v - np.max(v))  # 防止数值溢出
    e_x = e_x / np.sum(e_x, axis=0)
    for num,key in enumerate(keys):
        soft_dict[key] = e_x[num]
    return soft_dict




def sample_from_distribution(probabilities_dict):
    sorted_dict = sorted(probabilities_dict.items(), key=lambda x: x[1])
    rand_num = np.random.random()  # 生成[0, 1)之间的随机数
    cumulative_prob = 0.0
    for k, v in sorted_dict:
        cumulative_prob += v
        if rand_num <= cumulative_prob:
            return k