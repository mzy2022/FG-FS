import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,f_regression,chi2
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from Own.own_logger import error
import lightgbm as lgb
from Own.feature_eng.feature_cluster import cluster_features

# lgb.set_config(show_warning=False)
# lgb.set_config(show_info=False)
"""
这个脚本用来选择K个最好的特征
"""

def feature_selection(df,target,method_dict,cluster_dict,k_num):
    """
    选择K个特征
    :param df: dataframe or numpy
    :param target: list or dataframe
    :param method_dict: 字典，包括特征选择的方法，任务的类型：reg or clf
    :return: X_new: 新的特征集，selected_features: 选择特征的编号
    """
    selection_method = method_dict['method']
    task_type = method_dict['task_type']
    if selection_method is None:
        selection_method = 'SelectBest'
    if selection_method == 'SelectBest':
        score_funcs = {'reg': f_regression,'cls': mutual_info_classif}
        score_func = score_funcs[task_type]
        k_best = k_num
        selector = SelectKBest(score_func=score_func, k=k_best)
        X_new = selector.fit_transform(df, target)
        selected_features = selector.get_support(indices=True)
    elif selection_method == 'MutualInfo':
        score_funcs = {'reg': mutual_info_regression, 'cls': mutual_info_classif}
        score_func = score_funcs[task_type]
        k_best = method_dict['k_best']
        selector = SelectKBest(score_func=score_func, k=k_best)
        X_new = selector.fit_transform(df, target)
        selected_features = selector.get_support(indices=True)
    elif selection_method == 'DecisionTree':
        k_best = method_dict['k_best']
        estimators = {
            'reg': DecisionTreeRegressor(),
            'cls': DecisionTreeClassifier()
        }
        score_func = task_type
        est = estimators.get(score_func, DecisionTreeRegressor())
        selector = SelectFromModel(estimator=est, max_features=k_best)
        X_new = selector.fit_transform(df, target)
        selected_features = selector.get_support(indices=True)
    elif selection_method == 'l1':
        k_best = method_dict['k_best']
        estimators = {
            'cls': LogisticRegression(penalty='l1', solver='liblinear'),
            'reg': LinearRegression()
        }
        est = estimators[task_type]
        selector = SelectFromModel(estimator=est,max_features=k_best)
        X_new = selector.fit_transform(df, target)
        selected_features = selector.get_support(indices=True)
    else:
        error('wrong task name!!!!!')
        assert False

    new_dict = {key: [value for value in values if value in selected_features] for key, values in cluster_dict.items()}

    new_dist = {v: i for i, v in enumerate(sorted(set(value for values in new_dict.values() for value in values)))}
    sorted_dict = {k: [new_dist[value] for value in v] for k, v in new_dict.items()}

    return  pd.DataFrame(X_new),sorted_dict



def feature_selection_list(df,target,ori_name_list,task_type):
    """
    选择K个特征
    :param df: numpy
    :param target: list or dataframe
    :param method_dict: 字典，包括特征选择的方法，任务的类型：reg or clf
    :return: X_new: 新的特征集，selected_features: 选择特征的编号
    """
    task_type = task_type
    score_funcs = {'reg': f_regression,'cls': mutual_info_classif}
    score_func = score_funcs[task_type]
    if df.shape[1] > 50:
        k_best = 50
    else:
        k_best = df.shape[1]
    selector = SelectKBest(score_func=score_func, k=k_best)
    X_new = selector.fit_transform(df, target)
    selected_features = selector.get_support(indices=True)
    selected_features = list(selected_features)
    new_name = [ori_name_list[index] for index in selected_features]
    X_new = [row for row in X_new.T]
    return X_new, new_name


def feature_selection_new(df,target,num_features,task_type):
    feature_importance = get_feature_importances(df,target,task_type=task_type)
    sorted_list = sorted(enumerate(feature_importance), key=lambda x: x[1])
    original_order = [index for index, _ in sorted_list]
    selected_features = original_order[:num_features]
    return df.iloc[:,selected_features]


def feature_selection_new_ppo(df,target,num_features,cluster_dict,task_type):
    feature_importance = get_feature_importances(df, target,task_type=task_type)
    sorted_list = sorted(enumerate(feature_importance), key=lambda x: x[1])
    original_order = [index for index, _ in sorted_list]
    selected_features = original_order[:num_features]
    new_dict = {key: [value for value in values if value in selected_features] for key, values in cluster_dict.items()}
    return df.iloc[:, selected_features],new_dict

def get_feature_importances(X, y, estimator='avg', random_state=0, sample_count=1, sample_size=3, n_jobs=1,task_type='cls'):
    """Return feature importances by specifeid method """
    n_rows = X.shape[0]
    importance_sum = np.zeros(X.shape[1])
    total_estimators = []
    X = X.values
    for sampled in range(sample_count):
        sampled_ind = np.random.choice(np.arange(n_rows), size=n_rows // sample_size, replace=False)
        sampled_X = X[sampled_ind]
        sampled_y = np.take(y, sampled_ind)
        if estimator == "rf":
            if task_type == 'cls':
                estm = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
            else:
                estm = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
            estm.fit(sampled_X, sampled_y)
            total_importances = estm.feature_importances_
            estimators = estm.estimators_
            total_estimators += estimators
        elif estimator == "avg":
            if task_type == 'cls':
                clf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
            else:
                clf = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
            clf.fit(sampled_X, sampled_y)
            rf_importances = clf.feature_importances_
            estimators = clf.estimators_
            total_estimators += estimators
            train_data = lgb.Dataset(sampled_X, label=sampled_y)
            param = {'num_leaves': 31, 'objective': 'binary',  'verbosity': -1}
            param['metric'] = 'auc'
            num_round = 2
            bst = lgb.train(param, train_data, num_round)
            lgb_imps = bst.feature_importance(importance_type='gain')
            lgb_imps /= lgb_imps.sum()
            total_importances = (rf_importances + lgb_imps) / 2
        importance_sum += total_importances
        importance_sum = list(importance_sum)
    return importance_sum





# df = pd.read_csv('Openml_616.csv')
# X = df.iloc[:,:-1]
# y = df.iloc[:,-1]
# method_dict = {'method':'SelectBest','k_best':10,'task_type':'reg'}
# f_cluster = cluster_features(X)
# a, m = feature_selection_list(X.values,y,X.columns,task_type='reg')
# print(m)
