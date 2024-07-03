# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class Policy(nn.Module):
#     def __init__(self, args, data_nums,feature_nums, select_feature_nums, d_model, alpha, device):
#         super(Policy, self).__init__()
#         self.args = args
#         self.alpha = alpha
#         self.reduction_dimension = nn.Linear(data_nums, d_model)
#         self.layernorm = nn.LayerNorm(data_nums)
#         self.layernorm1 = nn.LayerNorm(feature_nums)
#         self.feature_nums = feature_nums
#         self.device = device
#         # self.multi_head_x = nn.Linear(d_model, 192)
#         # self.multi_head_y = nn.Linear(d_model, 192)
#         self.weight_x = nn.Parameter(torch.randn(1, d_model))
#         nn.init.xavier_uniform_(self.weight_x.data, gain=1.414)
#         self.weight_xy = nn.Parameter(torch.randn(1, d_model))
#         nn.init.xavier_uniform_(self.weight_xy.data, gain=1.414)
#         self.softmax = nn.Softmax(dim=-1)
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#         self.dropout = 0.1
#         self.sigmoid = nn.Sigmoid()
#         # self.mask =
#     def forward(self, input_x, input_y, adj_matrix):
#         # TODO:多头注意力
#         input_x_norm = self.layernorm(input_x)
#         input_y_norm = self.layernorm(input_y)
#         emb_data_x = self.reduction_dimension(input_x_norm)
#         # input_x_norm = self.mask * input_x_norm
#         emb_data_y = self.reduction_dimension(input_y_norm)
#         adj_matrix = torch.tensor(adj_matrix).to(self.device)
#         ### x attention
#         e = self.leakyrelu(torch.matmul(torch.matmul(emb_data_x, torch.diag(self.weight_x.squeeze())), emb_data_x.t()))
#         # e = torch.matmul(torch.matmul(emb_data_x, torch.diag(self.weight_x.squeeze())), emb_data_x.t())
#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj_matrix > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=0)
#         # attention = F.dropout(attention, self.dropout, training=True)
#         emb_data_x_prime = torch.matmul(attention, emb_data_x)
#
#         ### xy attention
#         e_xy = self.leakyrelu(torch.matmul(torch.matmul(emb_data_x, torch.diag(self.weight_xy.squeeze())), emb_data_y.t())).squeeze()
#         # e_xy = torch.matmul(torch.matmul(emb_data_x, torch.diag(self.weight_xy.squeeze())), emb_data_y.t()).squeeze()
#
#         if len(e_xy.shape) == 2:
#             e_xy = torch.sum(e_xy,dim=1)
#         attention_xy = F.softmax(e_xy, dim=0)
#         # attention_xy = F.dropout(attention_xy, self.dropout, training=True)
#         emb_data_xy_prime = attention_xy.unsqueeze(1) * emb_data_x
#
#         z = emb_data_x_prime + self.args.l * emb_data_xy_prime
#         mm = self.sigmod(z)
#         mm = self.layernorm1(torch.sum(z,dim=1,keepdim=True).squeeze())
#
#         z = F.softmax(mm)
#         # kkk = z.detach().cpu().numpy()
#         return z


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class FNN(nn.Module):
    def __init__(self, args, data_nums,feature_nums, select_feature_nums, d_model, alpha, device):
        super(FNN, self).__init__()
        self.args = args
        self.alpha = alpha
        self.reduction_dimension = nn.Linear(data_nums, d_model)
        self.layernorm = nn.LayerNorm(data_nums)
        self.feature_nums = feature_nums
        self.device = device

    def forward(self, input_x, input_y, adj_matrix):
        # TODO:多头注意力
        input_x_norm = self.layernorm(input_x)
        input_y_norm = self.layernorm(input_y)
        emb_data_x = self.reduction_dimension(input_x_norm)
        emb_data_y = self.reduction_dimension(input_y_norm)
        return emb_data_x, emb_data_y

class GAT(nn.Module):
    def __init__(self, args, data_nums,feature_nums, select_feature_nums, d_model, alpha, device):
        super(GAT, self).__init__()
        self.args = args
        self.alpha = alpha
        self.layernorm1 = nn.LayerNorm(feature_nums)
        self.feature_nums = feature_nums
        self.device = device
        self.multi_head_x = nn.Linear(d_model, 32* 6)
        self.multi_head_y = nn.Linear(d_model, 32 * 6)
        # self.multi_head_xy = nn.Linear(d_model, 32 *6)
        self.weight_x = nn.Parameter(torch.randn(1, 32))

        nn.init.xavier_uniform_(self.weight_x.data, gain=1.414)
        self.weight_xy = nn.Parameter(torch.randn(1, 32))
        nn.init.xavier_uniform_(self.weight_xy.data, gain=1.414)
        self.softmax = nn.Softmax(dim=-1)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.sigmoid = nn.Sigmoid()
        self.dropout = 0.1

    def forward(self, input_x, input_y, adj_matrix):
        emb_data_x = input_x
        emb_data_y = input_y
        adj_matrix = torch.tensor(adj_matrix).to(self.device)
        ### x attention
        emb_data_x = self.multi_head_x(emb_data_x).view(6,-1,32)
        e = torch.zeros((emb_data_x.shape[1], emb_data_x.shape[1])).to(self.device)
        for num,_ in enumerate(emb_data_x):
            emb_data = emb_data_x[num,:,:]
            matrix = self.leakyrelu(torch.matmul(torch.matmul(emb_data, torch.diag(self.weight_x.squeeze())), emb_data.t()))
            e += matrix

        e = e / 6
        # e = self.leakyrelu(torch.matmul(torch.matmul(emb_data_x, torch.diag(self.weight_x.squeeze())), emb_data_x.t()))
        # e = torch.matmul(torch.matmul(emb_data_x, torch.diag(self.weight_x.squeeze())), emb_data_x.t())
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)
        attention = F.softmax(attention, dim=0)
        # attention = F.dropout(attention, self.dropout, training=True)
        emb_data_x_prime = torch.matmul(attention, emb_data_x)

        emb_data_y = self.multi_head_y(emb_data_y).view(6, -1, 32)
        e_xy = torch.zeros(emb_data_x.shape[1]).to(self.device)
        for num, _ in enumerate(emb_data_y):
            emb_data_x_1 = emb_data_x[num, :, :]
            emb_data_y_1 = emb_data_y[num, :, :]
            matrix = self.leakyrelu(torch.matmul(torch.matmul(emb_data_x_1, torch.diag(self.weight_xy.squeeze())), emb_data_y_1.t())).squeeze()
            if len(matrix.shape) == 2:
                matrix = torch.sum(matrix, dim=1)
            e_xy += matrix

        e_xy = e_xy / 6
        ### xy attention
        # e_xy = self.leakyrelu(torch.matmul(torch.matmul(emb_data_x, torch.diag(self.weight_xy.squeeze())), emb_data_y.t())).squeeze()
        # e_xy = torch.matmul(torch.matmul(emb_data_x, torch.diag(self.weight_xy.squeeze())), emb_data_y.t()).squeeze()

        # if len(e_xy.shape) == 2:
        #     e_xy = torch.sum(e_xy,dim=1)
        attention_xy = F.softmax(e_xy, dim=0)
        # attention_xy = F.dropout(attention_xy, self.dropout, training=True)
        emb_data_xy_prime = attention_xy.unsqueeze(1) * emb_data_x

        z = emb_data_x_prime + self.args.l * emb_data_xy_prime
        z = z.view(emb_data_x_prime.shape[1],-1)
        mm = self.layernorm1(torch.sum(z,dim=1,keepdim=True).squeeze())
        mm = self.sigmoid(mm)
        # www = mm.detach().cpu().numpy()

        z = F.softmax(mm)
        # kkk = z.detach().cpu().numpy()
        return z
