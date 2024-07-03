import argparse

import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.distributions import Categorical
import torch.nn as nn
import torch
from Own.feature_eng.feature_cluster import cluster_features, cluster_features_1
from Own.feature_eng.feature_computation import O1, O2
from tools import feature_state_generation_des
from attention_worker import unary_Worker, binary_Worker
from Own.embedding_df.cov_emb import emb_features
from Own.Evolutionary_FE.update_data import update_binary, update_unary
from Own.feature_eng.feature_selection import feature_selection_new
torch.set_num_threads(5)


class Controller(nn.Module):
    def __init__(self, args, cluster_dict, process_data, ori_data):
        """

               :param args:
               :param cluster_dict: 由原始数据构成的聚类
               :param process_data: 不包括label的dataframe
               :param ori_data: 包括label的dataframe
               """
        super(Controller, self).__init__()
        self.args = args
        self.task_type = args.task_type
        self.hidden_size = 300
        self.embedding_size = 300
        self.device = torch.device('cuda',
                                   args.cuda) if torch.cuda.is_available() and args.cuda != -1 else torch.device('cpu')
        # self.continuous_col = args.continuous_col
        # self.discrete_col = args.discrete_col
        self.cluster_num = len(cluster_dict)
        self.cluster_dict = cluster_dict
        self.process_data = process_data
        self.ori_data = ori_data
        self.label = ori_data.iloc[:, -1]
        if self.task_type == 'cls':
            label_encoder = LabelEncoder()
            self.label = label_encoder.fit_transform(self.label)
            self.label = pd.Series(self.label)
        self.unary_type = O1
        self.binary_type = O2
        self.len_unary_type = len(O1)
        self.len_binary_type = len(O2)
        self.change_df = feature_state_generation_des
        self.len_change_df = 64
        self.step_num = 3
        self.num_layers = 2

        # RNN
        #  LSTMCell
        self.unary_rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers)
        #  LSTMCell
        self.binary_rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers)
        # unary operation embedding
        self.unary_decoder = nn.Linear(self.hidden_size, self.len_unary_type)
        # binary operation embedding
        self.binary_decoder = nn.Linear(self.hidden_size, self.len_binary_type)
        # attention
        self.binary_attention = nn.Linear(self.hidden_size, 1)
        self.unary_attention = nn.Linear(self.hidden_size, 1)
        self.binary_fc = nn.Linear(self.hidden_size, self.len_binary_type)
        self.unary_fc = nn.Linear(self.hidden_size, self.len_unary_type)

        self.init_unary_h = nn.Parameter(torch.zeros(self.num_layers, self.hidden_size))
        self.init_unary_c = nn.Parameter(torch.zeros(self.num_layers, self.hidden_size))

        self.init_binary_h = nn.Parameter(torch.ones(self.num_layers, self.hidden_size))
        self.init_binary_c = nn.Parameter(torch.ones(self.num_layers, self.hidden_size))
        self.init_parameters()

    def forward(self, input_data, h_c_t_list, is_unary):
        # LSTM + Attention
        if is_unary:
            output, hn = self.unary_rnn(input_data, h_c_t_list)
            output = output.unsqueeze(0)
            attention_weights = F.softmax(self.unary_attention(output), dim=1)
            attended_out = attention_weights * output
            res_h_c_list = hn
            unary_logits = self.unary_decoder(attended_out)
            return res_h_c_list, unary_logits

        else:
            output, hn = self.binary_rnn(input_data, h_c_t_list)
            output = output.unsqueeze(0)
            attention_weights = F.softmax(self.binary_attention(output), dim=1)
            attended_out = attention_weights * output
            res_h_c_list = hn
            binary_logits = self.binary_decoder(attended_out)
            return res_h_c_list, binary_logits

    def sample(self, is_unary, args):
        worker_u = unary_Worker(args)
        worker_b = binary_Worker(args)
        action_indexs_u = []
        states_u = []
        actions_u = []
        prob_vecs_u = []
        action_entropys_u = []
        emb_data_u = []
        action_indexs_b = []
        states_b = []
        actions_b = []
        prob_vecs_b = []
        action_entropys_b = []
        emb_data_b = []
        ori_states_u = []
        ori_states_b = []
        cluster_dict_u = []
        cluster_dict_b = []
        if is_unary:
            init_data = self.process_data.copy()
            cluster_dict = self.cluster_dict
            hc_unary_init = self.init_rnn_hidden(is_unary=True)
            emb_input = emb_features(args, init_data, cluster_dict, self.hidden_size)  # shape:(聚类数量，隐藏层数)
            for i in range(self.step_num):
                input_data = emb_input.to(self.device)
                res_h_c_list, unary_logits = self.forward(input_data, hc_unary_init, is_unary=True)
                # decoder-->softmaxa-->sample to actions_id
                unary_info = self.sample_cycle_component(unary_logits, 'unary')
                action_index_local = [unary_info['action_index_local']]
                prob_vec = [unary_info['prob_vec']]
                action_entropy = [unary_info['action_entropy']]
                unary_actions_name = unary_info['action_name']
                hc_unary_init = res_h_c_list
                propose_data = update_unary(cluster_dict, init_data, unary_actions_name, self.label, self.task_type)
                propose_cluster_dict = cluster_features(propose_data, self.label)
                emb_input = emb_features(args, propose_data, propose_cluster_dict, self.hidden_size)
                emb_data_u.append(input_data)
                states_u.append(propose_data)
                actions_u.append(unary_actions_name)
                action_indexs_u.append(action_index_local)
                prob_vecs_u.append(prob_vec)
                action_entropys_u.append(action_entropy)
                ori_states_u.append(init_data)
                cluster_dict_u.append(cluster_dict)

                init_data = propose_data
                cluster_dict = propose_cluster_dict

            worker_u.emb_data_u = emb_data_u
            worker_u.states_u = states_u
            worker_u.actions_u = actions_u
            worker_u.action_indexs_u = action_indexs_u
            worker_u.prob_vecs_u = prob_vecs_u
            worker_u.action_entropys_u = action_entropys_u
            worker_u.ori_states_u = ori_states_u
            worker_u.cluster_dict_u = cluster_dict_u


        else:
            init_data = self.process_data.copy()
            cluster_dict = self.cluster_dict
            hc_binary_init = self.init_rnn_hidden(is_unary=False)
            emb_input = emb_features(args, init_data, cluster_dict, self.hidden_size)  # shape:(聚类数量，隐藏层数)
            for i in range(self.step_num):
                input_data = emb_input.to(self.device)
                res_h_c_list, binary_logits = self.forward(input_data, hc_binary_init, is_unary=False)
                # decoder-->softmaxa-->sample to actions_id
                binary_info = self.sample_cycle_component(binary_logits, 'binary')
                action_index_local = [binary_info['action_index_local']]
                prob_vec = [binary_info['prob_vec']]
                action_entropy = [binary_info['action_entropy']]
                binary_actions_name = binary_info['action_name']
                hc_binary_init = res_h_c_list
                #  根据采样的操作符更新原dataframe
                propose_data = update_binary(cluster_dict, init_data, binary_actions_name, self.label, self.task_type)
                if propose_data.shape[1] > 2 * init_data.shape[1]:
                    propose_data = feature_selection_new(propose_data,self.label,num_features=int(1.5 * init_data.shape[1]),task_type=self.task_type)
                propose_cluster_dict = cluster_features(propose_data, self.label)
                emb_input = emb_features(args, propose_data, propose_cluster_dict, self.hidden_size)
                #  更新workers
                emb_data_b.append(input_data)
                states_b.append(propose_data)
                actions_b.append(binary_actions_name)
                action_indexs_b.append(action_index_local)
                prob_vecs_b.append(prob_vec)
                action_entropys_b.append(action_entropy)
                ori_states_b.append(init_data)
                cluster_dict_b.append(cluster_dict)

                init_data = propose_data
                cluster_dict = propose_cluster_dict

            worker_b.emb_data_b = emb_data_b
            worker_b.states_b = states_b
            worker_b.actions_b = actions_b
            worker_b.action_indexs_b = action_indexs_b
            worker_b.prob_vecs_b = prob_vecs_b
            worker_b.action_entropys_b = action_entropys_b
            worker_b.ori_states_b = ori_states_b
            worker_b.cluster_dict_b = cluster_dict_b

        if is_unary:
            return worker_u
        else:
            return worker_b

    def sample_cycle_component(self, logits, output_type):
        action_index_local = Categorical(logits=logits).sample()
        prob_matrix = F.softmax(logits, dim=2).squeeze(0)
        log_prob_matrix = F.log_softmax(logits, dim=2).squeeze(0)

        reshape_action_index = action_index_local.reshape(-1, 1)

        prob_vec = torch.gather(
            prob_matrix, dim=1,
            index=reshape_action_index).reshape(1, -1)

        log_prob_vec = torch.gather(
            log_prob_matrix, dim=1,
            index=reshape_action_index).reshape(1, -1)

        action_entropy = -1 * torch.sum(prob_vec * log_prob_vec)

        if output_type == 'unary':
            action_name = [self.unary_type[int(i)] for i in action_index_local[0]]
        elif output_type == 'binary':
            action_name = [self.binary_type[int(i)] for i in action_index_local[0]]
        else:
            action_name = None
        return {
            'ops_type': output_type,
            'prob_vec': prob_vec.detach(),
            'action_entropy': action_entropy.detach(),
            'action_index_local': action_index_local.detach(),
            'action_name': action_name
        }

    ###PPO new_一元操作更新
    def unary_new_prob(self, workers_u, detach_bool=False, args=None):
        worker_u = unary_Worker(args)
        action_entropy_u = []
        prob_vec_u = []
        worker = workers_u
        hc_unary_init = self.init_rnn_hidden(is_unary=True)
        states = worker.ori_states_u
        cluster_dicts = worker.cluster_dict_u
        for index, state in enumerate(states):
            emb_input = emb_features(args, state, cluster_dicts[index], self.hidden_size)  # shape:(聚类数量，隐藏层数)
            input_data = emb_input.to(self.device)
            res_h_c_list, unary_logits = self.forward(input_data, hc_unary_init, is_unary=True)
            unary_info = self.prod_cycle_component(unary_logits)
            prob_vec = unary_info['prob_vec']
            action_entropy = unary_info['action_entropy'].reshape(-1)
            if detach_bool:
                action_entropy_u.append(action_entropy.detach())
                prob_vec_u.append(prob_vec.detach())
            else:
                action_entropy_u.append(action_entropy)
                prob_vec_u.append(prob_vec)
            hc_unary_init = res_h_c_list
        worker_u.action_entropys_u = action_entropy_u
        worker_u.prob_vecs_u = prob_vec_u
        return worker_u

    ###PPO new_二元操作更新
    def binary_new_prob(self, worker_binary, detach_bool=False, args=None):
        worker_b = binary_Worker(args)
        action_entropy_b = []
        prob_vec_b = []
        worker = worker_binary
        hc_binary_init = self.init_rnn_hidden(is_unary=False)
        states = worker.ori_states_b
        cluster_dicts = worker.cluster_dict_b
        for index, state in enumerate(states):
            emb_input = emb_features(args, state, cluster_dicts[index],
                                     self.hidden_size)  # shape:(聚类数量，隐藏层数)
            input_data = emb_input.to(self.device)
            res_h_c_list, binary_logits = self.forward(input_data, hc_binary_init, is_unary=False)
            binary_info = self.prod_cycle_component(binary_logits)
            prob_vec = binary_info['prob_vec']
            action_entropy = binary_info['action_entropy'].reshape(-1)
            if detach_bool:
                action_entropy_b.append(action_entropy.detach())
                prob_vec_b.append(prob_vec.detach())
            else:
                action_entropy_b.append(action_entropy)
                prob_vec_b.append(prob_vec)
            hc_binary_init = res_h_c_list
        worker_b.action_entropys_b = action_entropy_b
        worker_b.prob_vecs_b = prob_vec_b
        return worker_b

    @staticmethod
    def prod_cycle_component(logits):
        action_index_local = Categorical(logits=logits).sample()
        prob_matrix = F.softmax(logits, dim=2).squeeze(0)
        log_prob_matrix = F.log_softmax(logits, dim=2).squeeze(0)

        reshape_action_index = action_index_local.reshape(-1, 1)

        prob_vec = torch.gather(
            prob_matrix, dim=1,
            index=reshape_action_index).reshape(1, -1)

        log_prob_vec = torch.gather(
            log_prob_matrix, dim=1,
            index=reshape_action_index).reshape(1, -1)

        action_entropy = -1 * torch.sum(prob_vec * log_prob_vec)
        return {
            'prob_vec': prob_vec,
            'action_entropy': action_entropy
        }

    def init_rnn_hidden(self, is_unary):
        state_size = (self.num_layers, self.hidden_size)
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
