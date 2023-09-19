import numpy as np
import pandas as pd
from Roulette_Wheel_Selection import wheel_selection
from Own.feature_eng.feature_computation import O1,justify_operation_type


"""
特征变异操作，对每一个特征聚类进行变异，返回新的特征集
"""

def mutation(f_cluster,label,op,f_cluster_name,task_type):
    """

    :param f_cluster: dataframe
    :param label: 标签，dataframe
    :param op: 操作符
    :param f_cluster_name: dataframe的列名
    :param label_name: 标签名称
    :param task_type: 任务类型
    :return:
    """
    mutation_dict = wheel_selection(f_cluster,label,task_type)
    assert op in O1
    op_sign = justify_operation_type(op)
    f_new, f_new_name = [], []
    for k,v in mutation_dict.items():
        cluster_data = f_cluster.iloc[:,v].values             #cluster_data 为ndarray
        if op == 'sqrt':
            for i in range(cluster_data.shape[1]):
                if np.sum(cluster_data[:, i] < 0) == 0:
                    f_new.append(op_sign(cluster_data[:, i]))
                    f_new_name.append(str(f_cluster_name[v[i]]) + '_' + str(op))
            # f_generate = np.array(f_new).T
            # if len(f_generate) != 0:
            #     feas = f_generate
            #     feas_name = f_new_name
        elif op == 'reciprocal':
            for i in range(cluster_data.shape[1]):
                if np.sum(cluster_data[:, i] == 0) == 0:
                    f_new.append(op_sign(cluster_data[:, i]))
                    f_new_name.append(str(f_cluster_name[v[i]]) + '_' + str(op))
            # f_generate = np.array(f_new).T
            # if len(f_generate) != 0:
            #     feas = f_generate
            #     feas_name = f_new_name
        elif op == 'log':
            for i in range(cluster_data.shape[1]):
                if np.sum(cluster_data[:, i] <= 0) == 0:
                    f_new.append(op_sign(cluster_data[:, i]))
                    f_new_name.append(str(f_cluster_name[v[i]]) + '_' + str(op))
            # f_generate = np.array(f_new).T
            # if len(f_generate) != 0:
            #     feas = f_generate
            #     feas_name = f_new_name
        elif op == 'none':
            for i in range(cluster_data.shape[1]):
                f_new.append(cluster_data[:, i])
                f_new_name.append(str(f_cluster_name[v[i]]))
        else:
            for i in range(cluster_data.shape[1]):
                f_new.append(op_sign(cluster_data[:,i]))
                f_new_name.append(str(f_cluster_name[v[i]]) + '_' + str(op))
    f_generate = np.array(f_new).T
    df = pd.DataFrame(f_generate,columns=f_new_name)
    return df


def cube(x):
    return x ** 3

def inverse(x):
    '''
        :type col: list or np.array
        :rtype: np.array,shape = (len(array),1)
        '''
    try:
        return 1 / x if x != 0 else 0
    except:
        raise ValueError('Value type error,check feature type')


data = pd.read_csv("hepatitis.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_name = X.columns
df = mutation(X,y,'none',X_name,'cls')
print(df)