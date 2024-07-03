import lightgbm
import pandas as pd
import featuretools as ft
import xgboost
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
# 假设你有一个数据框
# data = {'user_id': [1, 2, 3, 4],
#         'age': [23, 34, 45, 56],
#         'income': [50000, 60000, 70000, 80000]}
# dataframe = pd.DataFrame(data)
def sub_rae(y, y_hat):
    y = np.array(y).reshape(-1)
    y_hat = np.array(y_hat).reshape(-1)
    y_mean = np.mean(y)

    # 计算每个预测值与实际值之间的绝对误差
    absolute_errors = np.abs(y_hat - y)

    # 计算每个实际值与均值之间的绝对误差
    mean_errors = np.abs(y_mean - y)

    # 计算RAE，然后计算其补值
    rae = np.sum(absolute_errors) / np.sum(mean_errors)
    res = 1 - rae

    return res
name = 'PimaIndian'
path = fr"D:/python files/pythonProject3/DQN_my2/data/{name}.csv"
data = pd.read_csv(path)

data['new_column'] = range(len(data))
columns = ['new_column'] + [col for col in data.columns if col != 'new_column']
df = data[columns]
trans_primitives = ['add_numeric', 'subtract_numeric', 'multiply_numeric', 'divide_numeric']
agg_primitives = ['sum', 'median','mean']
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# 创建 EntitySet
es = ft.EntitySet(id='my_dataset')

# 将数据框添加到 EntitySet 中，设置 user_id 作为 index
es = es.add_dataframe(dataframe_name='my',
                      dataframe=X,
                      index='new_column')

# 使用 dfs 生成特征
features, feature_names = ft.dfs(entityset=es,
                                 target_dataframe_name='my',
                                 agg_primitives=agg_primitives,
                                 trans_primitives=trans_primitives,
                                 max_depth=2,
                                 verbose=1)

features[features > 1e15] = 0
features[features < -1e15] = 0
features = features.apply(np.nan_to_num)
features = features.replace([np.inf, -np.inf], 0)
# clf = RandomForestClassifier(n_estimators=10, random_state=0)

scores = cross_val_score(clf, features, y, scoring='f1_micro', cv=5)
# model = RandomForestRegressor(n_estimators=10, random_state=0)
# rae_score1 = make_scorer(sub_rae, greater_is_better=True)
# scores = cross_val_score(model, features, y, cv=5, scoring=rae_score1)
# 查看生成的特征
print(np.mean(scores))



# 加载数据
# data = pd.read_csv('test.csv')
# data['new_column'] = range(len(data))
# columns = ['new_column'] + [col for col in data.columns if col != 'new_column']
# df = data[columns]
# # 定义实体集
# es = ft.EntitySet(id='ecommerce')
# es = es.add_dataframe(dataframe_name='purchases',
#                       dataframe=data,
#                       index='new_column',  # 假设我们有一个购买ID作为主键
# )  # 购买日期作为时间索引
# agg_primitives = ['sum', 'mean', 'count', 'min', 'max']
# # 使用DFS生成特征
# # 在这个例子中，我们设置max_depth为1，意味着我们将生成基于单个字段的直接特征，而不涉及字段的组合
# features, feature_names = ft.dfs(entityset=es,
#                                  target_dataframe_name ='purchases',
#                                  agg_primitives=agg_primitives,
#                                  max_depth=2,
#                                  verbose=1)  # verbose=1将打印出正在生成的特征

# 查看生成的特征
# print(features.head())


