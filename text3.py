import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adj_matrix, feature_matrix):
        output = torch.matmul(adj_matrix, feature_matrix)  # 图卷积操作
        output = self.linear(output)  # 线性变换
        output = F.relu(output)  # ReLU 激活函数
        return output


class GCN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        self.num_nodes = num_nodes

    def forward(self, adj_matrix, feature_matrix):
        # 第一层图卷积
        hidden = self.gcn1(adj_matrix, feature_matrix)
        # 第二层图卷积
        output = self.gcn2(adj_matrix, hidden)
        return output


# 定义一个简单的图结构和特征矩阵
adj_matrix = torch.tensor([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=torch.float32)
feature_matrix = torch.tensor([[1, 0],
                               [0, 1],
                               [1, 1]], dtype=torch.float32)

# 创建 GCN 模型
num_nodes = adj_matrix.shape[0]
input_dim = feature_matrix.shape[1]
hidden_dim = 16
output_dim = 8
model = GCN(10, 2, hidden_dim, output_dim)

# 将图结构和特征矩阵传入模型进行前向计算
output = model(adj_matrix, feature_matrix)
print("Output shape:", output.shape)
