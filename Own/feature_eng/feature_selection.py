import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,f_regression,chi2
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from Own.own_logger import error

"""
这个脚本用来选择K个最好的特征
"""

def feature_selection(df,target,method_dict):
    """
    选择K个特征
    :param df: dataframe or numpy
    :param target: list or dataframe
    :param method_dict: 字典，包括特征选择的方法，任务的类型：reg or clf
    :return: X_new: 新的特征集，selected_features: 选择特征的编号
    """
    selection_method = method_dict['method']
    task_type = method_dict['task_type']
    if selection_method == None:
        selection_method = 'SelectBest'
    if selection_method == 'SelectBest':
        score_funcs = {'reg': f_regression,'clf': chi2}
        score_func = score_funcs[task_type]
        k_best = method_dict['k_best']
        selector = SelectKBest(score_func=score_func, k=k_best)
        X_new = selector.fit_transform(df, target)
        selected_features = selector.get_support(indices=True)
    elif selection_method == 'MutualInfo':
        score_funcs = {'reg': mutual_info_regression, 'clf': mutual_info_classif}
        score_func = score_funcs[task_type]
        k_best = method_dict['k_best']
        selector = SelectKBest(score_func=score_func, k=k_best)
        X_new = selector.fit_transform(df, target)
        selected_features = selector.get_support(indices=True)
    elif selection_method == 'DecisionTree':
        k_best = method_dict['k_best']
        estimators = {
            'reg': DecisionTreeRegressor(),
            'clf': DecisionTreeClassifier()
        }
        score_func = task_type
        est = estimators.get(score_func, DecisionTreeRegressor())
        selector = SelectFromModel(estimator=est, max_features=k_best)
        X_new = selector.fit_transform(df, target)
        selected_features = selector.get_support(indices=True)
    elif selection_method == 'l1':
        k_best = method_dict['k_best']
        estimators = {
            'clf': LogisticRegression(penalty='l1', solver='liblinear'),
            'reg': LinearRegression()
        }
        est = estimators[task_type]
        selector = SelectFromModel(estimator=est,max_features=k_best)
        X_new = selector.fit_transform(df, target)
        selected_features = selector.get_support(indices=True)
    else:
        error('wrong task name!!!!!')
        assert False
    return  X_new,selected_features


data = {'Feature1': [1, 4, 7],
        'Feature2': [2, 5, 8],
        'Feature3': [3, 6, 9],
        'A':[2,4,7],
        'b':[1,3,7]}
df = pd.read_csv('Openml_616.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
method_dict = {'method':'SelectBest','k_best':10,'task_type':'reg'}
a, m = feature_selection(X,y,method_dict)
print(m)
