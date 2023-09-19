import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


# 定义回归模型
class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    # 加载数据并进行划分


data = pd.read_csv('wine_red.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
unique_labels = np.unique(y)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train = torch.tensor([label_map[label] for label in y_train], dtype=torch.long)
y_test = torch.tensor([label_map[label] for label in y_test], dtype=torch.long)


# 创建模型并定义损失函数和优化器
model = RegressionModel(X.shape[1], len(np.unique(y)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)
# y_test = torch.tensor(y_test, dtype=torch.long)
# 训练模型
for epoch in range(1000):
    model.train()  # 将模型设为训练模式
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    # _, predicted_train = torch.max(y_pred.data, 1)
    # train_accuracy = accuracy_score(y_train.numpy(), predicted_train.numpy())
    # model.eval()
    # with torch.no_grad():
    #     y_logits_test = model(X_test)
    #     _, predicted_test = torch.max(y_logits_test.data, 1)
    #     test_accuracy = accuracy_score(y_test.numpy(), predicted_test.numpy())
    #
    # print(
    #     f'Epoch [{epoch + 1}/{100}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

# 在测试集上进行预测
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    _, predicted = torch.max(y_pred.data, 1)

# 计算评估指标（例如RMSE）
accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
print('Accuracy:', accuracy)