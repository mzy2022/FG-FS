import torch.nn as nn
import torch
from .modules_rl import StatisticLearning, EncoderLayer, SelectOperations, ReductionDimension, weight_init,PoswiseFeedForwardNet
from torch.distributions.categorical import Categorical
import logging, os
# from Own.ppo_attention.emedding_data.main_emedding_data import train_experiment
# from Own.ppo_attention.emb_data2 import modules
from Own.ppo_attention.emb_data_tab import tab_transformer_pytorch
class Actor(nn.Module):
    def __init__(self, args, data_nums, data_features,operations, d_model, d_k, d_v, d_ff, n_heads, dropout=None, enc_load_pth=None):
        super(Actor, self).__init__()
        self.args = args
        self.reduction_dimension = ReductionDimension(data_nums, d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        logging.info(f"Randomly initial encoder")
        if os.path.exists(enc_load_pth):
            self.encoder.load_state_dict(torch.load(enc_load_pth))
            logging.info(f"Successfully load encoder, enc_load_pth:{enc_load_pth}")
        self.select_operation = SelectOperations(d_model, operations)
        self.c_nums = len(args.c_columns)
        self.layernorm = nn.LayerNorm(data_nums)
        self.model = tab_transformer_pytorch.TabTransformer(
            n_samples=data_nums,
            emb_dim = 8,
            dim=128,
            depth=1,
            heads=4,
            attn_dropout=0.1,  # post-attention dropout
            ff_dropout=0.1,  # feed forward dropout
        )



    def forward(self, input_num,input_col, h,c,step):
        res_out = self.model(input_num, input_col)
        enc_outputs = self.pos_ffn(res_out)
        enc_outputs = enc_outputs.squeeze(0)
        output,h,c= self.select_operation(enc_outputs,h,c)
        # output,h,c = self.select_operation(enc_outputs,h,c)
        output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)
        operation_softmax = torch.softmax(output, dim=-1)

        return operation_softmax,h,c
