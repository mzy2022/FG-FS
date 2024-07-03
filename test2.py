import random

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# def rf_classify():
#     model = RandomForestClassifier(n_estimators=10, random_state=0)
#     return model
#
# data = pd.read_csv('test.csv')
# data2 = pd.read_csv('train.csv')
# X = data.iloc[:,:-1]
# y = data.iloc[:,-1]
# X2 = data2.iloc[:,:-1]
# y2 = data2.iloc[:,-1]
# def cross(X, y):
#     # if self.metric == '1-rae':
#     #     return cross_val_score(self.eval_model.model, X, y, scoring=make_scorer(one_minus_rae), cv=5).mean()
#     # X.columns = X.columns.astype(str)
#     return cross_val_score(rf_classify(), X, y, scoring='f1_micro', cv=5).mean()
#
# a = cross(X,y)
# b = cross(X2,y2)
# print(a)
# print(b)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
# MNIST数据集的加载和预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    for data in trainloader:
        img, _ = data
        img = img.view(img.size(0), -1).cuda()
        output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# 提取编码器权重
encoder_weights = model.encoder[0].weight.data.cpu().numpy()

# 计算每个特征的重要性
feature_importance = np.mean(np.abs(encoder_weights), axis=0)

# 选择最重要的100个像素点
num_features = 100
selected_features = np.argsort(feature_importance)[-num_features:]

# 显示选择的像素点
mask = np.zeros(784)
mask[selected_features] = 1
plt.imshow(mask.reshape(28, 28), cmap='gray')
plt.show()


# 重新加载数据，仅使用选择的特征
def select_features(data, selected_features):
    return data.view(data.size(0), -1)[:, selected_features]


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


# 模型、损失函数和优化器
input_dim = num_features
num_classes = 10
model = LogisticRegressionModel(input_dim, num_classes).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练分类器
num_epochs = 20
for epoch in range(num_epochs):
    for data in trainloader:
        images, labels = data
        images = select_features(images.cuda(), selected_features)

        outputs = model(images)
        loss = criterion(outputs, labels.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估分类器
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = select_features(images.cuda(), selected_features)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

