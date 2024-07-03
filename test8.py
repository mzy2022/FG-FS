# import torch
# def get_adj_matrix(feature_matrix,cosine):
#     feature_matrix = torch.tensor(feature_matrix)
#     norms = torch.norm(feature_matrix, dim=0, keepdim=True)
#
#     # 归一化特征向量
#     normalized_features = feature_matrix / norms
#     adj_list = []
#     # 计算归一化后的特征向量的点积，即余弦相似度矩阵
#     cosine_similarity_matrix = torch.mm(normalized_features.t(), normalized_features)
#     for i in range(0,len(cosine_similarity_matrix)):
#         for j in range(i+1,len(cosine_similarity_matrix)):
#             adj_list.append(cosine_similarity_matrix[i][j])
#     adj_list = sorted(adj_list,reverse=True)
#     x = int(cosine * len(adj_list))
#     threshold = adj_list[x]
#     cosine_similarity_matrix[cosine_similarity_matrix <= threshold] = 0
#
#
#     return cosine_similarity_matrix
#
# feature_matrix = torch.tensor([[1.0, -1.0, 3.0],
#                                [4.0, -4.0, 6.0],
#                                [-4.0, 4.0, 6.0],
#                                [2,-2,4]])
# x = get_adj_matrix(feature_matrix,0.8)
# print(x)