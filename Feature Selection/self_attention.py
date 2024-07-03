import torch.nn as nn
import torch
import numpy as np


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

        # outputs: [b_size x n_heads x len_q x d_v]
        context = torch.matmul(attn, v)

        return context, attn

class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
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
    def __init__(self, d_model, d_k, d_v, n_heads, dropout=True):
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


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads, dropout=None):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        # self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask=None):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        # enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs

class Attention(nn.Module):
    def __init__(self, data_nums, d_model, d_k, d_v, d_ff, n_heads, dropout=None):
        super(Attention, self).__init__()
        self.reduction_dimension = ReductionDimension(data_nums, d_model)
        self.layernorm = nn.LayerNorm(data_nums)
        self.encoder = EncoderLayer(d_model, d_k, d_v, d_ff, n_heads, dropout)
        self.decoder = nn.Linear(d_model,data_nums)

    def forward(self, input):
        input_norm = self.layernorm(input)
        data_reduction_dimension = self.reduction_dimension(input_norm)
        data_reduction_dimension = torch.where(torch.isnan(data_reduction_dimension),
                                               torch.full_like(data_reduction_dimension, 0), data_reduction_dimension)
        encoder_output = self.encoder(data_reduction_dimension)
        encoder_output = torch.where(torch.isnan(encoder_output), torch.full_like(encoder_output, 0), encoder_output).squeeze(0)
        encoder_output = self.decoder(encoder_output)

        return encoder_output

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
