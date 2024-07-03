import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv('data/lymphography.csv')
X = np.array(data.iloc[:,:-1])
y = np.array(data.iloc[:,-1])
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 计算每个特征与目标变量之间的互信息
# mi = mutual_info_classif(X_train, y_train)

# 打印每个特征的互信息值


# 选择互信息最高的K个特征
k = 10  # 假设我们选择前2个最重要的特征
selector = SelectKBest(mutual_info_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 打印所选特征的索引


# 训练分类器并评估性能（示例使用随机森林分类器）
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train_selected, y_train)
y_pred = clf.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"使用选择的{k}个特征进行分类的准确率：{accuracy}")
