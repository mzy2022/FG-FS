import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import logging, os
import numpy as np


class Actor(nn.Module):
    def __init__(self, args, data_nums,feature_nums, operations_c, operations_d, d_model, d_k, d_v, d_ff, n_heads, rnn_hidden_size,
                 dropout=None, enc_load_pth=None):
        super(Actor, self).__init__()
        self.args = args
        self.reduction_dimension = ReductionDimension(data_nums, d_model)
        self.encoder = EncoderLayer(d_model, d_k, d_v, d_ff, n_heads, dropout)
        self.decoder = DecoderLayer(d_model, d_k, d_v, d_ff, n_heads, dropout)
        self.select_operation = SelectOperations(d_model,feature_nums, rnn_hidden_size, operations_c)
        self.layernorm = nn.LayerNorm(data_nums)
        self.feature_nums = feature_nums

    def forward(self, input, step, res_h_c_t_list):

        if step == 0:
            h_c_t_list = self.select_operation.init_rnn_hidden()
        else:
            h_c_t_list = res_h_c_t_list
        input_norm = self.layernorm(input)
        data_reduction_dimension = self.reduction_dimension(input_norm)
        data_reduction_dimension = torch.where(torch.isnan(data_reduction_dimension),
                                               torch.full_like(data_reduction_dimension, 0), data_reduction_dimension)
        encoder_output = self.encoder(data_reduction_dimension)
        encoder_output = torch.where(torch.isnan(encoder_output), torch.full_like(encoder_output, 0),
                                     encoder_output).squeeze(0)
        # encoder_output = data_reduction_dimension.squeeze(0)
        encoder_mean = self.mean(encoder_output, self.feature_nums)
        res_h_c_t_list, ops_logits = self.select_operation(encoder_mean, h_c_t_list)
        ops_logits = torch.where(torch.isnan(ops_logits), torch.full_like(ops_logits, 0), ops_logits)
        ops_softmax = torch.softmax(ops_logits, dim=-1)

        return ops_softmax, res_h_c_t_list

    def mean(self,encoder_output,feature_nums):
        num_groups = len(encoder_output) // feature_nums
        result = []
        for i in range(feature_nums):
            if i != feature_nums - 1:
                group = encoder_output[i * num_groups: (i + 1) * num_groups]
            else:
                group = encoder_output[i * num_groups: ]
            group_avg = torch.mean(group,dim=0)
            result.append(group_avg)
        result = torch.stack(result)
        return result

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class SelectOperations(nn.Module):
    def __init__(self, d_model, feature_nums,hidden_size, num_ops):
        self.hidden_size = hidden_size
        super().__init__()
        #  LSTMCell
        self.ops_rnn = nn.LSTMCell(d_model, hidden_size)
        # M operation embedding
        # T operation embedding
        self.ops_decoder = nn.Linear(hidden_size, num_ops)
        self.feature_nums = feature_nums
        state_size = (self.feature_nums, hidden_size)
        self.init_ops_h = nn.Parameter(torch.zeros(state_size))
        self.init_ops_c = nn.Parameter(torch.zeros(state_size))

        self.init_parameters()

    def init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

        self.ops_decoder.bias.data.fill_(0)

    def init_rnn_hidden(self):
        state_size = (self.feature_nums, self.hidden_size)
        init_ops_h = self.init_ops_h.expand(*state_size).contiguous()
        init_ops_c = self.init_ops_c.expand(*state_size).contiguous()
        h_c_t_list = [(init_ops_h, init_ops_c)]
        return h_c_t_list

    def forward(self, input_data, h_c_t_list):

        ops_h_t, ops_c_t = h_c_t_list[0]

        ops_h_t, ops_c_t = self.ops_rnn(input_data, (ops_h_t, ops_c_t))
        ops_logits = self.ops_decoder(ops_h_t)

        res_h_c_t_list = [(ops_h_t, ops_c_t)]
        return res_h_c_t_list, ops_logits


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=None):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_sign = dropout
        if self.dropout_sign:
            self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        if self.dropout_sign:
            attn = self.dropout(self.softmax(scores))
        else:
            attn = self.softmax(scores)
        context = torch.matmul(attn, v)
        return context, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        # d_model = 128,d_k=d_v = 32,n_heads = 4
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_k * n_heads)
        self.w_k = nn.Linear(d_model, d_k * n_heads)
        self.w_v = nn.Linear(d_model, d_v * n_heads)

        self.attention = ScaledDotProductAttention(d_k, dropout)

    def forward(self, q, k, v, attn_mask):
        b_size = q.size(0)
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout=None):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multihead_attn = _MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.proj = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_sign = dropout
        if self.dropout_sign:
            self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, q, k, v, attn_mask=None):
        residual = q
        context, attn = self.multihead_attn(q, k, v, attn_mask=attn_mask)
        context = torch.where(torch.isnan(context), torch.full_like(context, 0), context)
        attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)
        if self.dropout_sign:
            output = self.dropout(self.proj(context))
        else:
            output = self.proj(context)

        ro = residual + output
        no = self.layer_norm(ro)
        if torch.isnan(no).any() and not torch.isnan(ro).any():
            return ro, attn
        return no, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=None):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_sign = dropout
        if self.dropout_sign:
            self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        if self.dropout_sign:
            output = self.dropout(output)

        return self.layer_norm(residual + output)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads, dropout=None):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask=None):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads, dropout=None):
        super(DecoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask=None):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs


class ReductionDimension(nn.Module):
    def __init__(self, statistic_nums, d_model):
        super(ReductionDimension, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(statistic_nums),
            nn.Linear(statistic_nums, d_model),
            nn.BatchNorm1d(d_model),
        )

    def forward(self, input):
        out = self.layer(input).unsqueeze(dim=0)
        return out
