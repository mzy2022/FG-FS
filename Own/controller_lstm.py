import argparse

import pandas as pd
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn as nn
import torch

from Own.feature_eng.feature_cluster import cluster_features
from Own.feature_eng.feature_computation import O1, O2
from tools import feature_state_generation_des

torch.set_num_threads(5)


class Controller(nn.Module):
    def __init__(self, args, cluster_dict, process_data,ori_data):
        """

        :param args:
        :param cluster_dict: 由原始数据构成的聚类
        :param process_data: 不包括label的dataframe
        :param ori_data: 包括label的dataframe
        """
        super(Controller, self).__init__()
        self.args = args
        self.hidden_size = 300
        self.embedding_size = 4
        self.device = torch.device('cuda',
                                   args.cuda) if torch.cuda.is_available() and args.cuda != -1 else torch.device('cpu')
        # self.continuous_col = args.continuous_col
        # self.discrete_col = args.discrete_col
        self.cluster_num = len(cluster_dict)
        self.cluster_dict = cluster_dict
        self.process_data = process_data
        self.ori_data = ori_data
        self.unary_type = O1
        self.binary_type = O2
        self.len_unary_type = len(O1)
        self.len_binary_type = len(O2)
        self.unary_embedding_actions = self.unary_type + ['PADDING']
        self.binary_embedding_actions = self.binary_type + ['PADDING']
        self.change_df = feature_state_generation_des
        self.len_change_df = 64
        self.step_num = 5
        # Embedding
        self.unary_embedding = nn.Embedding(len(self.unary_embedding_actions), self.embedding_size)
        self.binary_embedding = nn.Embedding(len(self.binary_embedding_actions), self.embedding_size)
        # RNN
        #  LSTMCell
        self.unary_rnn = nn.LSTMCell(self.embedding_size, self.hidden_size)
        #  LSTMCell
        self.binary_rnn = nn.LSTMCell( self.embedding_size, self.hidden_size)
        # unary operation embedding
        self.unary_decoder = nn.Linear(self.hidden_size, self.len_unary_type)
        # binary operation embedding
        self.binary_decoder = nn.Linear(self.hidden_size, self.len_binary_type)

        state_size = (1, self.hidden_size)
        self.init_unary_h = nn.Parameter(torch.zeros(state_size))
        self.init_unary_c = nn.Parameter(torch.zeros(state_size))

        self.init_binary_h = nn.Parameter(torch.ones(state_size))
        self.init_binary_c = nn.Parameter(torch.ones(state_size))
        self.init_parameters()


    def forward(self, input_data, h_c_t_list, is_unary):

        if is_unary:
            # RNN
            # input_data = torch.LongTensor(input_data).unsqueeze(0).to(self.device)

            input_data = self.unary_embedding(input_data)
            input_data = input_data.reshape(-1, self.embedding_size)
            unary_h_t, unary_c_t = self.unary_rnn(input_data, h_c_t_list)
            unary_logits = self.unary_decoder(unary_h_t)
            res_h_c_list = (unary_h_t, unary_c_t)
            return res_h_c_list, unary_logits
        else:
            input_data = self.binary_embedding(input_data)
            input_data = input_data.reshape(-1, self.embedding_size)
            binary_h_t, binary_c_t = self.binary_rnn(input_data, h_c_t_list)
            binary_logits = self.binary_decoder(binary_h_t)
            res_h_c_list = (binary_h_t, binary_c_t)
            return res_h_c_list, binary_logits

    def sample(self, is_unary):
        input_data = []
        if is_unary:
            hc_unary_init = self.init_rnn_hidden(is_unary=True)
            unary_sample_history = []
            for i in range(self.step_num):
                sample_history = []
                for pos in range(self.cluster_num):
                    if len(input_data) == 0:
                        input_data = ['PADDING']
                        input_data = [self.unary_embedding_actions.index(i) for i in input_data]
                        input_data = torch.LongTensor(input_data).unsqueeze(0).to(self.device)
                    else:
                        input_data = action_index_local[0].unsqueeze(0).to(self.device)
                    res_h_c_list, unary_logits = self.forward(input_data, hc_unary_init, is_unary=True)
                    # decoder-->softmaxa-->sample to actions_id
                    unary_info = self.sample_cycle_component(unary_logits, 'unary')
                    action_index_local = [unary_info['action_index_local']]
                    prob_vec = [unary_info['prob_vec']]
                    action_entropy = [unary_info['action_entropy']]
                    unary_actions_name = unary_info['action_name']
                    action_index_global = unary_info['action_index_global']
                    sample_history.append({
                        'action_index_local': action_index_local,
                        'unary_actions_name': unary_actions_name,
                        'action_entropy': action_entropy,
                        'prob_vec': prob_vec,
                        'unary_logits': unary_logits.detach(),
                        'action_index_global':action_index_global
                    })
                    input_data = unary_logits
                unary_sample_history.append((i,sample_history))
                hc_unary_init = res_h_c_list
                action_index_global = []
                for i in sample_history:
                    num = int(i['action_index_global'])
                    op = self.unary_embedding_actions[num]
                    action_index_global.append(op)
                input_data = action_index_global
                input_data = [self.unary_embedding_actions.index(i) for i in input_data]
                input_data = torch.LongTensor(input_data).unsqueeze(0).to(self.device)

        else:
            hc_binary_init = self.init_rnn_hidden(is_unary=False)
            binary_sample_history = []
            for i in range(self.step_num):
                sample_history = []
                for pos in range(self.cluster_num):
                    if len(input_data) == 0:
                        input_data = ['PADDING']
                        input_data = [self.binary_embedding_actions.index(i) for i in input_data]
                        input_data = torch.LongTensor(input_data).unsqueeze(0).to(self.device)
                    else:
                        input_data = action_index_local[0].unsqueeze(0).to(self.device)

                    res_h_c_list, binary_logits = self.forward(input_data, hc_binary_init, is_unary=False)

                    # decoder-->softmaxa-->sample to actions_id
                    binary_info = self.sample_cycle_component(binary_logits, 'binary')
                    action_index_local = [binary_info['action_index_local']]
                    prob_vec = [binary_info['prob_vec']]
                    action_entropy = [binary_info['action_entropy']]
                    binary_actions_name = binary_info['action_name']
                    action_index_global = binary_info['action_index_global']
                    sample_history.append({
                        'action_index_local': action_index_local,
                        'binary_actions_name': binary_actions_name,
                        'action_entropy': action_entropy,
                        'prob_vec': prob_vec,
                        'binary_logits': binary_logits.detach(),
                        'action_index_global': action_index_global
                    })
                    input_data = binary_logits
                binary_sample_history.append((i, sample_history))
                hc_binary_init = res_h_c_list
                action_index_global = []
                for i in sample_history:
                    num = int(i['action_index_global'])
                    op = self.binary_embedding_actions[num]
                    action_index_global.append(op)
                input_data = action_index_global
                input_data = [self.binary_embedding_actions.index(i) for i in input_data]
                input_data = torch.LongTensor(input_data).unsqueeze(0).to(self.device)

        return_dict = dict()
        if is_unary:
            return_dict['unary_feature_engineering'] = unary_sample_history
        else:
            return_dict['binary_feature_engineering'] = binary_sample_history

        return return_dict

    def sample_cycle_component(self, logits, output_type):
        global action_index_global
        action_index_local = Categorical(logits=logits).sample()
        prob_matrix = F.softmax(logits, dim=1)
        log_prob_matrix = F.log_softmax(logits, dim=1)

        prob_vec = prob_matrix[0][int(action_index_local)]
        log_prob_vec = log_prob_matrix[0][int(action_index_local)]

        action_entropy = -1 * torch.sum(prob_vec * log_prob_vec)

        if output_type == 'unary':
            action_name = self.unary_type[action_index_local]
            action_index_global = [self.unary_embedding_actions.index(action_name)]
            action_index_global = torch.LongTensor(action_index_global)
        elif output_type == 'binary':
            action_name = self.binary_type[action_index_local]
            action_index_global = [self.binary_embedding_actions.index(action_name)]
            action_index_global = torch.LongTensor(action_index_global)
        else:
            action_name = None
        return {
            'action_index_global': action_index_global,
            'ops_type': output_type,
            'prob_vec': prob_vec.detach(),
            'action_entropy': action_entropy.detach(),
            'action_index_local': action_index_local.detach(),
            'action_name': action_name,
        }

    def get_prod(self,sample_history,is_unary,detach_bool=False):
        if not is_unary:
            input_data = ['PADDING'] * self.cluster_num
            input_data = [self.binary_embedding_actions.index(i) for i in input_data]
            input_data = torch.LongTensor(input_data).unsqueeze(0).to(self.device)
            sample_history_fe = sample_history['binary_feature_engineering']
            hc_list = self.init_rnn_hidden(is_unary=False)
            prod_history = []
            for sample_index, sampled_his in enumerate(sample_history_fe):
                ###二元操作采样
                res_h_c_t_list, binary_logits = self.forward(input_data, hc_list,is_unary=False)
                action_index_local = sampled_his[1]['action_index_local']
                binary_info = self.prod_cycle_component(binary_logits, action_index_local)
                prob_vec = binary_info['prob_vec']
                action_entropy = binary_info['action_entropy'].reshape(-1)
                if detach_bool:
                    prod_history.append({
                        'action_entropy': action_entropy.detach(),
                        'prob_vec': prob_vec.detach()
                    })
                else:
                    prod_history.append({
                        'action_entropy': action_entropy,
                        'prob_vec': prob_vec
                    })
                hc_list = res_h_c_t_list
        else:
            input_data = ['PADDING']
            input_data = [self.unary_embedding_actions.index(i) for i in input_data]
            input_data = torch.LongTensor(input_data).unsqueeze(0).to(self.device)
            sample_history_fe = sample_history['unary_feature_engineering']
            hc_list = self.init_rnn_hidden(is_unary=False)
            prod_history = []

            # 一元操作采样
            for sample_index, sampled_his in enumerate(sample_history_fe):
                res_h_c_t_list, unary_logits = self.forward(input_data, hc_list, is_unary=True)
                for fe_his in sampled_his[1]:
                    action_index_local = fe_his['action_index_local']
                    unary_info = self.prod_cycle_component(unary_logits, action_index_local)
                    prob_vec = unary_info['prob_vec']
                    action_entropy = unary_info['action_entropy'].reshape(-1)
                    if detach_bool:
                        prod_history.append({
                            'action_entropy': action_entropy.detach(),
                            'prob_vec': prob_vec.detach()
                        })
                    else:
                        prod_history.append({
                            'action_entropy': action_entropy,
                            'prob_vec': prob_vec
                        })
                    hc_list = res_h_c_t_list
        return_dict = dict()
        return_dict['feature_engineering'] = prod_history
        return return_dict

    @staticmethod
    def prod_cycle_component(logits, action_index_local):
        prob_matrix = F.softmax(logits, dim=1)
        log_prob_matrix = F.log_softmax(logits, dim=1)
        prob_vec = prob_matrix[0][int(action_index_local[0])]
        log_prob_vec = log_prob_matrix[0][int(action_index_local[0])]
        action_entropy = -1 * torch.sum(prob_vec * log_prob_vec)
        return {
            'prob_vec': prob_vec,
            'action_entropy': action_entropy
        }

    def init_rnn_hidden(self, is_unary):
        state_size = (1, self.hidden_size)
        if is_unary:
            init_unary_h = self.init_unary_h.expand(*state_size).contiguous()
            init_unary_c = self.init_unary_c.expand(*state_size).contiguous()
            return (init_unary_h, init_unary_c)
        else:
            init_binary_h = self.init_binary_h.expand(*state_size).contiguous()
            init_binary_c = self.init_binary_c.expand(*state_size).contiguous()
            return (init_binary_h, init_binary_c)

    def init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

        self.binary_decoder.bias.data.fill_(0)
        self.unary_decoder.bias.data.fill_(0)



# def init_param():
#     parser = argparse.ArgumentParser(description='PyTorch Experiment')
#     parser.add_argument('--name', type=str, default='wine_red', help='data name')
#     parser.add_argument('--data', type=str, default="winequality_red", help='dataset name')
#     parser.add_argument('--model', type=str, default="rf")
#     parser.add_argument('--cuda', type=int, default=-1, help='which gpu to use')  # -1 represent cpu-only
#     parser.add_argument('--coreset', type=int, default=0,
#                         help='whether to use coreset')  # 1 represent work with coreset
#     parser.add_argument('--core_size', type=int, default=10000, help='size of coreset')  # m-->sample size
#     parser.add_argument('--psm', type=int, default=1,
#                         help='whether to use policy-set-merge')  # >0 represent work with psm, and the value eauals the number of Policy-set
#     parser.add_argument('--is_select', type=int, default=50)
#     parser.add_argument('--seed', type=int, default=0, help='random seed')
#
#     args = parser.parse_args()
#
#     return args

# data = pd.read_csv("hepatitis.csv")
# X = data.iloc[:,:-1]
# y = data.iloc[:,-1]
# X_name = X.columns
# args = init_param()
# dis = cluster_features(X)
# controller = Controller(args, dis, X,X)
# x = controller.sample(is_unary=False)
# m = controller.get_prod(x,is_unary=False)
# print(m)
