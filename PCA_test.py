import lightgbm
import numpy as np
import pandas as pd
import xgboost
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
path = f"D:/python files/pythonProject3/DQN_my2/data/PimaIndian.csv"


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


# 加载数据集
data = pd.read_csv(path)
X = data.iloc[:,:-1]  # 特征
y = data.iloc[:,-1]  # 标签

# 将数据分成训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建LDA模型
m = np.unique(y).size-1
fe = min(X.shape[1],m)
pca = PCA(n_components=fe)  # 设置LDA降维后的维度数量

# 对训练数据进行LDA降维
X = pca.fit_transform(X, y)
clf = RandomForestClassifier(n_estimators=10, random_state=0)

# 创建流水线，将LDA和分类器组合在一起
pipeline = make_pipeline(pca, clf)
# 使用交叉验证评估模型
scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_micro')


# model = RandomForestRegressor(n_estimators=10, random_state=0)
# pipeline = make_pipeline(pca, model)
# rae_score1 = make_scorer(sub_rae, greater_is_better=True)
# scores = cross_val_score(pipeline, X, y, cv=5, scoring=rae_score1)
# 创建分类器模型


# 在LDA降维后的特征上训练模
print("分类准确度:", np.mean(scores))
