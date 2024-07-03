import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor1(nn.Module):
    def __init__(self, args, data_nums, feature_nums,d_model, d_k,n_heads,alpha,device):
        super(Actor1, self).__init__()
        self.args = args
        self.alpha = alpha
        self.reduction_dimension = ReductionDimension(data_nums, d_model)
        self.eval_net = QNet_ops(d_model, 2)
        self.target_net = QNet_ops(d_model, 2)
        self.layernorm = nn.LayerNorm(data_nums)
        self.feature_nums = feature_nums
        self.device = device
        self.multi_head_x = nn.Linear(d_model, d_k * n_heads)
        self.multi_head_y = nn.Linear(d_model, d_k * n_heads)
        self.multi_head_xy = nn.Linear(d_model, d_k * n_heads)
        self.weight_x = nn.Parameter(torch.randn(1, d_model))
        self.weight_y = nn.Parameter(torch.randn(1, d_model))
        self.weight_xy = nn.Parameter(torch.randn(1, d_model))
        self.ffn_x = nn.Linear(d_model, d_model)
        self.ffn_y = nn.Linear(d_model, d_model)
        self.ffn_xy = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_x, input_y, for_next):
        global z
        input_x_norm = self.layernorm(input_x)
        input_y_norm = self.layernorm(input_y)
        data_x = self.reduction_dimension(input_x_norm)
        data_y = self.reduction_dimension(input_y_norm)
        emb_data_x = torch.where(torch.isnan(data_x), torch.full_like(data_x, 0), data_x)
        emb_data_y = torch.where(torch.isnan(data_y), torch.full_like(data_y, 0), data_y)
        for i in range(4):

            if i == 0:
                z = torch.concat([emb_data_x, emb_data_y], dim=0)
                z = torch.mean(z,dim=0)

            elif i == 1:
                a_list = []
                emb_x = []
                for j in range(emb_data_x.shape[0]):
                    for k in range(j + 1, emb_data_x.shape[0]):
                        parameter_tensor = self.weight_x.data
                        diagonal_matrix = torch.diag(parameter_tensor.squeeze())
                        x = (emb_data_x[j] @ diagonal_matrix @ emb_data_x[k].t()).item()
                        similarity = self.get_Cosine(emb_data_x[j], emb_data_x[k])
                        if torch.sum(similarity) < self.alpha:
                            x = 0.0
                        a_list.append(x)
                        emb_x.append(emb_data_x[j] + emb_data_x[k])
                if len(a_list) == 0:
                    a_list.append(1.0)
                if len(emb_x) != 0:
                    emb_data_x = torch.stack(emb_x).to(self.device)
                a_list = torch.tensor(a_list).to(self.device)
                if len(emb_data_x) == 0:
                    r = z
                else:
                    r = self.softmax(a_list) @ emb_data_x + z
                z = self.ffn_x(r) + r

            elif i == 2:
                a_list = []
                emb_data_xy = []
                for j in range(emb_data_x.shape[0]):
                    for k in range(emb_data_y.shape[0]):
                        parameter_tensor = self.weight_xy.data
                        diagonal_matrix = torch.diag(parameter_tensor.squeeze())
                        kkk = emb_data_x[j].unsqueeze(0)
                        mmmmm = emb_data_y[k].unsqueeze(0).t()
                        x = (kkk @ diagonal_matrix @ mmmmm).item()
                        similarity = self.get_Cosine(emb_data_x[j], emb_data_y[k])
                        mmm = torch.sum(similarity)
                        if mmm < self.alpha:
                            x = 0.0
                        a_list.append(x)
                        emb_data_xy.append(emb_data_x[j] + emb_data_y[k])
                if len(emb_data_xy) != 0:
                    emb_data_xy = torch.stack(emb_data_xy).to(self.device)
                    a_list = torch.tensor(a_list).to(self.device)
                    r = self.softmax(a_list) @ emb_data_xy + z
                else:
                    r = z
                z = self.ffn_x(r) + r

            elif i == 3:
                a_list = []
                emb_y = []
                for j in range(emb_data_y.shape[0]):
                    for k in range(j + 1, emb_data_y.shape[0]):
                        parameter_tensor = self.weight_y.data
                        diagonal_matrix = torch.diag(parameter_tensor.squeeze())
                        x = (emb_data_y[j] @ diagonal_matrix @ emb_data_y[k].t()).item()
                        similarity = self.get_Cosine(emb_data_y[j], emb_data_y[k])
                        if torch.sum(similarity) < self.alpha:
                            x = 0.0
                        a_list.append(x)
                        emb_y.append(emb_data_y[j] + emb_data_y[k])
                if len(a_list) == 0:
                    a_list.append(1.0)
                if len(emb_y) != 0:
                    emb_data_y = torch.stack(emb_y).to(self.device)
                    a_list = torch.tensor(a_list).to(self.device)
                    r = self.softmax(a_list) @ emb_data_y + z
                else:
                    r = z
                z = self.ffn_x(r) + r

        z = torch.where(torch.isnan(z), torch.full_like(z, 0), z).squeeze(0)
        if for_next:
            ops_logits = self.target_net(z)
        else:
            ops_logits = self.eval_net(z)
        ops_logits = torch.where(torch.isnan(ops_logits), torch.full_like(ops_logits, 0), ops_logits)
        return ops_logits, z

    def get_Cosine(self, x, y):
        return x.t() * y / torch.norm(x, p=2) / torch.norm(y, p=2)


class QNet_ops(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, init_w=0.1, device=None):
        super(QNet_ops, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.out = nn.Linear(hidden_dim, action_dim)
        self.out.weight.data.normal_(-init_w, init_w)
        self.device = device

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class ReductionDimension(nn.Module):
    def __init__(self, statistic_nums, d_model):
        super(ReductionDimension, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(statistic_nums, d_model),
        )

    def forward(self, input):
        out = self.layer(input)
        return out
