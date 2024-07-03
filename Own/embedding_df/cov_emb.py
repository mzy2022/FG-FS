import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch.distributions import Categorical

"""
为每个聚类生成embedding的tensor,维度为(1,hidden_size)
"""
class ReductionDimension(nn.Module):
    def __init__(self, num_samples,num_features, d_model):
        super(ReductionDimension, self).__init__()
        self.statistic_nums = num_samples
        self.d_model = d_model
        self.layer = nn.Sequential(
            # nn.BatchNorm1d(num_samples),
            nn.Linear(num_samples, d_model),
        )
        self.layer2 = nn.Linear(num_features * d_model, d_model)
# 归一化



    def forward(self, input):
        out = self.layer(input)
        out1 = out.reshape(-1,input.shape[0] * self.d_model)
        ao = self.layer2(out1)

        return ao


class Emb_df(nn.Module):
    def __init__(self, num_samples, num_features, d_model):
        super(Emb_df, self).__init__()
        self.reduction_dimension = ReductionDimension(num_samples,num_features, d_model)
        self.layernorm = nn.LayerNorm(num_samples)

    def forward(self, input_c):
        input = torch.from_numpy(input_c.values).float().transpose(0, 1)
        input_norm = self.layernorm(input)
        data_reduction_dimension = self.reduction_dimension(input_norm)
        data_reduction_dimension = torch.where(torch.isnan(data_reduction_dimension),
                                               torch.full_like(data_reduction_dimension, 0), data_reduction_dimension)
        return data_reduction_dimension


def emb_features(args,ori_df,cluster_dict,hidden_size):
    ff = []
    cluster_list = list(cluster_dict.items())
    for i in range(len(cluster_list)):
        v = cluster_list[i][1]
        pro_df = ori_df.iloc[:,v]
        num_samples = pro_df.shape[0]
        num_features = pro_df.shape[1]
        emb_df = Emb_df(num_samples,num_features,hidden_size)
        finish_emb = emb_df(pro_df)
        ff.append(finish_emb)
    my_tensor = torch.cat(ff)
    return my_tensor

# from Own.feature_eng.feature_cluster import cluster_features
# data = pd.read_csv('SPECTF.csv')
# X = data.iloc[:,:-1]
# y = data.iloc[:,-1]
# dict = cluster_features(X,y)
# args = None
# mmm = emb_features(args,X,dict,300)







# actor = Emb_df(267,135,128)
# data = pd.read_csv('SPECTF.csv')
# scaler = MinMaxScaler()
# df_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
# a = actor(df_normalized)
# a = a.reshape(4,32)
#
#
#
# unary_rnn = nn.LSTM(32, 200,2)
# h0 = torch.randn(2, 200)
# c0 = torch.randn(2, 200)
# output, hn = unary_rnn(a, (h0, c0))
#
# output = output.unsqueeze(0)
# attention = nn.Linear(200, 1)
# fc = nn.Linear(200, 5)
# attention_weights = F.softmax(attention(output), dim=1)
# attended_out = attention_weights * output
#
#
#
# ops_decoder = nn.Linear(200, 5)
# x = ops_decoder(attended_out)
# action_index_local = Categorical(logits=x).sample()
# prob_matrix = F.softmax(x, dim=1)
# log_prob_matrix = F.log_softmax(x, dim=1)
# print(attended_out)
