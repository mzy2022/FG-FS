# from copy import deepcopy
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset, Subset
# from torchmetrics import Accuracy, AUROC
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.datasets import load_iris
#
# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, dropout,output_size):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size, output_size))
#
#     def forward(self,x):
#         out = self.model(x)
#         return out
#
#
#     def train_MLP(self,model, criterion, train_loader,val_loader, num_epochs):
#         self.optimizer = optim.Adam(model.parameters(), lr=0.001)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,  factor=0.2, patience=2,min_lr=1e-6, verbose=True)
#           # 将模型设置为训练模式
#         for epoch in range(num_epochs):
#             model.train()
#             for inputs, targets in train_loader:
#                   # 梯度清零
#                 outputs = model(inputs)  # 前向传播
#                 loss = criterion(outputs, targets)  # 计算损失
#                 loss.backward()  # 反向传播
#                 self.optimizer.step()  # 更新参数
#                 self.optimizer.zero_grad()
#
#
#             model.eval()
#             with torch.no_grad():
#                 # For mean loss.
#                 pred_list = []
#                 label_list = []
#                 for inputs, targets in val_loader:
#                     outputs = model(inputs)
#                     pred_list.append(outputs.cpu())
#                     label_list.append(targets.cpu())
#                 y = torch.cat(label_list, 0)
#                 pred = torch.cat(pred_list, 0)
#                 val_loss = criterion(pred, y).item()
#                 scheduler.step(val_loss)
#
#                 if val_loss == scheduler.best:
#                     best_model = deepcopy(model)
#
#         self.restore_parameters(model, best_model)
#
#     def eetest(self,model,test_loader,metric,auroc):
#         pred_list = []
#         label_list = []
#         model.eval()
#         with torch.no_grad():
#             for inputs, targets in test_loader:
#                 outputs = model(inputs)
#                 pred_list.append(outputs.cpu())
#                 label_list.append(targets.cpu())
#         y = torch.cat(label_list, 0)
#         pred = torch.cat(pred_list, 0)
#         score = metric(pred, y).item()
#         auroc_score = auroc(pred, y).item()
#         print(f"分数：{score},AUROC:{auroc_score}")
#         return score
#
#
#
#
#     def restore_parameters(self,model, best_model):
#         '''Move parameters from best model to current model.'''
#         for param, best_param in zip(model.parameters(), best_model.parameters()):
#             param.data = best_param
#
#
# def data_split(dataset, val_portion=0.2, test_portion=0.2, random_state=0):
#     '''
#     Split dataset into train, val, test.
#
#     Args:
#       dataset: PyTorch dataset object.
#       val_portion: percentage of samples for validation.
#       test_portion: percentage of samples for testing.
#       random_state: random seed.
#     '''
#     # Shuffle sample indices.
#     rng = np.random.default_rng(random_state)
#     inds = np.arange(len(dataset))
#     rng.shuffle(inds)
#
#     # Assign indices to splits.
#     n_val = int(val_portion * len(dataset))
#     n_test = int(test_portion * len(dataset))
#     test_inds = inds[:n_test]
#     val_inds = inds[n_test:(n_test + n_val)]
#     train_inds = inds[(n_test + n_val):]
#
#     # Create split datasets.
#     test_dataset = Subset(dataset, test_inds)
#     val_dataset = Subset(dataset, val_inds)
#     train_dataset = Subset(dataset, train_inds)
#     return train_dataset, val_dataset, test_dataset
#
# def make_onehot(x):
#     '''Make an approximately one-hot vector one-hot.'''
#     argmax = torch.argmax(x, dim=1)
#     onehot = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
#     onehot[torch.arange(len(x)), argmax] = 1
#     # wwww = onehot.detach().cpu().numpy()
#     return onehot
#
# if __name__ == '__main__':
#     data = pd.read_csv("syn1.csv")
#
#
#
#     # 加载鸢尾花数据集作为示例
#
#     # 使用互信息作为评分函数选择前2个特征
#
#
#
#     hidden_size = 128  # 隐藏层大小
#
#     output_size = len(np.unique(data.iloc[:,-1]) ) # 输出类别数量（例如，MNIST有10个类别）
#     num_epochs = 250# 迭代次数
#     dropout = 0.3
#
#     # X = data.iloc[:, :-1]
#     # y = data.iloc[:, -1]
#     features = np.array([f for f in data.columns if f not in ['Outcome']])
#
#     x = np.array(data.drop(['Outcome'], axis=1)[features]).astype('float32')
#     y = np.array(data['Outcome']).astype('int64')
#     for i in range(1, 10):
#         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#         k_best = SelectKBest(score_func=mutual_info_classif, k=i)
#         X_selected = k_best.fit_transform(x_train, y_train)
#         # 获取所选特征的索引
#         selected_indices = k_best.get_support(indices=True)
#         x_new = x_train[:, selected_indices]
#
#         # 创建随机森林分类器模型
#         rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#         # 训练模型
#         rf_classifier.fit(x_new, y_train)
#
#         # 使用训练好的模型进行预测
#         y_pred = rf_classifier.predict(x_test[:,selected_indices])
#
#         # 计算准确率
#         accuracy = accuracy_score(y_test, y_pred)
#         print("随机森林模型的准确率:", accuracy)
#
#
#
#
#     for i in range(1,10):
#         k_best = SelectKBest(score_func=mutual_info_classif, k=i)
#         X_selected = k_best.fit_transform(x, y)
#         # 获取所选特征的索引
#         selected_indices = k_best.get_support(indices=True)
#         x_new = x[:,selected_indices]
#         input_size = x_new.shape[1] # 输入特征维度（例如，MNIST图像的大小为28x28，展开为784维向量）
#
#         data = TensorDataset(torch.from_numpy(x_new), torch.from_numpy(y))
#         data.features = features
#         data.input_size = x_new.shape[1]
#         data.output_size = len(np.unique(y))
#         train_dataset, val_dataset, test_dataset = data_split(data)
#         train_loader = DataLoader(train_dataset, batch_size=128,shuffle=True, pin_memory=True, drop_last=True)
#         val_loader = DataLoader(val_dataset, batch_size=1024, pin_memory=True)
#         test_loader = DataLoader(test_dataset, batch_size=1024, pin_memory=True)
#         #####################################
#         # data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
#         # x_train = np.array(data_train.drop(['label'], axis=1)[features]).astype('float32')
#         # y_train = np.array(data_train['label']).astype('int64')
#         #
#         # x_test = np.array(data_train.drop(['label'], axis=1)[features]).astype('float32')
#         # y_test = np.array(data_train['label']).astype('int64')
#         # # Create dataset object.
#         # data_train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
#         # data_train.features = features
#         # data_train.input_size = x_train.shape[1]
#         # data_train.output_size = len(np.unique(y_train))
#         #
#         # data_test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
#         # data_test.features = features
#         # data_test.input_size = x_test.shape[1]
#         # data_test.output_size = len(np.unique(y_test))
#         # train_loader = DataLoader(data_train, batch_size=128,shuffle=True, pin_memory=True, drop_last=True)
#         # test_loader = DataLoader(data_test, batch_size=1024, pin_memory=True)
#         #########################
#         # 构建模型、损失函数和优化器
#         model = MLP(input_size, hidden_size,dropout, output_size)
#         torch.save(model.state_dict(), 'model.pth')
#         criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
#          # 随机梯度下降优化器
#         metric = Accuracy(task='multiclass', num_classes=output_size)
#         auroc_metric = lambda pred, y: AUROC(task='multiclass', num_classes=output_size)(pred.softmax(dim=1), y)
#
#         # 训练模型
#         model.train_MLP(model, criterion, train_loader, val_loader,num_epochs)
#
#         model.eetest(model,test_loader,metric,auroc_metric)
#
#
import torch
import sys

# 检查Python版本
print("Python version")
print(sys.version)

# 检查PyTorch版本
print("PyTorch version")
print(torch.__version__)