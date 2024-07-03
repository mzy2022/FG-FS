# 奖励函数
# 节点覆盖率奖励：
# def node_coverage_reward(selected_nodes, original_graph):
#     covered_nodes = set(selected_nodes)
#     for node in selected_nodes:
#         covered_nodes.update(original_graph.neighbors(node))
#     coverage_rate = len(covered_nodes) / original_graph.number_of_nodes()
#     return coverage_rate
#
#
# # 边覆盖率奖励：
# def edge_coverage_reward(selected_nodes, original_graph):
#     covered_edges = 0
#     total_edges = original_graph.number_of_edges()
#     for node in selected_nodes:
#         for neighbor in original_graph.neighbors(node):
#             if neighbor in selected_nodes:
#                 covered_edges += 1
#     coverage_rate = covered_edges / total_edges
#     return coverage_rate
#
# # 保持度奖励（如聚类系数）：
# import networkx as nx
#
# def clustering_coefficient_reward(selected_nodes, original_graph):
#     subgraph = original_graph.subgraph(selected_nodes)
#     original_clustering = nx.average_clustering(original_graph)
#     subgraph_clustering = nx.average_clustering(subgraph)
#     reward = 1 - abs(original_clustering - subgraph_clustering)
#     return reward
#
#
# # 平均路径长度奖励：
# def average_path_length_reward(selected_nodes, original_graph):
#     subgraph = original_graph.subgraph(selected_nodes)
#     try:
#         original_apl = nx.average_shortest_path_length(original_graph)
#         subgraph_apl = nx.average_shortest_path_length(subgraph)
#         reward = 1 - abs(original_apl - subgraph_apl) / original_apl
#     except nx.NetworkXError:
#         reward = 0  # 如果子图不是连通的，奖励为0
#     return reward


import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

# 加载Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# data.x: 节点特征矩阵
print(data.x.shape)  # 输出: torch.Size([2708, 1433])
# data.edge_index: 边索引矩阵
print(data.edge_index.shape)  # 输出: torch.Size([2, 10556])
# num_features: 每个节点的特征维度
num_features = dataset.num_node_features
print(num_features)  # 输出: 1433
# num_classes: 类别数量
num_classes = dataset.num_classes
print(num_classes)  # 输出: 7

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.gat2 = GATConv(8*8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 初始化模型和优化器
model = GATNet(in_channels=num_features, out_channels=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 训练模型
for epoch in range(200):
    loss = train()
    print(f'Epoch {epoch}, Loss: {loss}')

# 获取节点的重要性评分
def get_node_importance_scores(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    scores = torch.norm(out, dim=1)
    return scores

# 选择代表节点
def select_representative_nodes(scores, k):
    _, topk_indices = torch.topk(scores, k)
    return topk_indices

scores = get_node_importance_scores(model, data)
k = 10  # 选择前10个代表节点
representative_nodes = select_representative_nodes(scores, k)
print("Representative nodes:", representative_nodes)



