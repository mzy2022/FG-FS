import argparse
import copy
import time

import pandas as pd
from tqdm import trange
from tqdm import tqdm
from Own.feature_eng.feature_selection import feature_selection,feature_selection_new_ppo
from Own.feature_eng.feature_cluster import cluster_features, cluster_features_1
from controller_lstm import Controller
import torch.optim as optim
from worker import unary_Worker,binary_Worker
import numpy as np
import torch
import logging
from reward import GetReward
from Own.Evolutionary_FE.Crossover import crossover
from Own.Evolutionary_FE.Mutation import mutation
from Own.feature_eng.data_trans import Pipeline_data


class PPO_psm(object):
    def __init__(self, args):
        self.base_ori_score = None
        self.clip_epsion = 0.2
        self.args = args
        self.arch_lr = 0.001
        self.episodes = 5
        self.entropy_weight = 1e-3
        self.policy_nums = 150
        self.task_type = args.task_type
        self.eval_method = args.eval_method
        self.final_action_ori = None
        self.ppo_epochs = 30
        self.device = torch.device('cuda',
                                   args.cuda) if torch.cuda.is_available() and args.cuda != -1 else torch.device('cpu')
        self.get_reward_ins = GetReward(self.args)
        self.baseline = None
        self.final_action = None
        self.baseline_weight = 0.9
        self.clip_epsilon = 0.2
        self.base_score_list = []
        self.base_ori_score_list = []
        self.base_ori_score_list = None
        self.ori_data = pd.read_csv(args.dataset_path)
        self.f_ori_data = self.ori_data.iloc[:,:-1]
        self.target = self.ori_data.iloc[:,-1]
        self.new_data = None
        self.max_len_features = 2 * self.ori_data.shape[1]
        self.binary = None
        self.best_binary_op = None
        self.best_unary_op = None
        self.new_data_pipeline = Pipeline_data(all_data=self.ori_data,continuous_columns=None,discrete_columns=None)

    def get_ori_base_score(self):
        score_list = self.get_reward_ins.downstream_task_new(self.f_ori_data,self.target, task_type=self.task_type)
        return score_list

    def feature_search(self):
        self.base_ori_score = self.get_ori_base_score()
        print(f"thrdfd:{self.base_ori_score}")
        self.new_data = []
        self.mas = []
        for k in range(self.policy_nums):
            print(f"policy_nums:{k}")
            best_score = self.base_ori_score
            binary_sample_workers = []
            unary_sample_workers = []
            ids_unary = []
            ids_binary = []
            ma = -1 * float("inf")
            self.process_data = copy.deepcopy(self.ori_data)
            self.target = self.process_data.iloc[:, -1]
            self.process_data = self.process_data.iloc[:, :-1]
            self.best_data = copy.deepcopy(self.process_data)
            # self.cluster_dict = wheel_selection(self.process_data,self.target,self.task_type)
            self.cluster_dict = cluster_features(self.process_data,self.target)

            #   根据新生成的process_data进行多轮采样二元操作集
            self.controller = Controller(self.args, self.cluster_dict, self.process_data, self.ori_data).to(self.device)
            for try_num in trange(self.episodes, desc='Episodes'):
                tmp = []
                sample_history = self.controller.sample(is_unary=False)
                binary_worker = binary_Worker(self.args, sample_history, is_uanry=False)
                tmp.append(binary_worker.actions_trans)
                tmp.append(try_num)
                ids_binary.append(tmp)
                binary_sample_workers.append(binary_worker)
            for i in tqdm(ids_binary, desc='Binary IDs'):
                print(f"ids_binary:{i}")
                tmp_op_list = i[0]
                start_time = time.time()
                new_feature_dict,new_cluster_dict = crossover(self.cluster_dict, self.process_data, self.target,
                                                                 tmp_op_list, self.process_data.columns,
                                                                 task_type=self.task_type)
                end_time = time.time()
                execution_time = end_time - start_time
                print("crossover代码执行时间：", execution_time, "秒")

                for j in range(len(new_feature_dict)):
                    self.process_data, self.cluster_dict = new_feature_dict[j],new_cluster_dict[j]
                    print(f'{self.process_data.shape}:process_data的大小')
                    start_time = time.time()
                    if len(self.process_data.columns) > self.max_len_features:
                        self.process_data,self.cluster_dict = feature_selection(self.process_data, self.target,
                                                              {'method': 'SelectBest', 'task_type': self.task_type},self.cluster_dict,self.max_len_features)
                        # self.process_data,self.cluster_dict = feature_selection_new_ppo(self.process_data,self.target,self.max_len_features,self.cluster_dict)


                    end_time = time.time()
                    execution_time = end_time - start_time
                    print("特征选择代码执行时间：", execution_time, "秒")
                    start_time = time.time()
                    score = self.get_reward_ins.downstream_task_new(self.process_data, self.target,task_type=self.task_type)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print("分数score代码执行时间：", execution_time, "秒")
                    print(f"score:{score},,,score----{best_score - score}")
                    if score > best_score:
                        best_score = score
                        self.base_ori_score = best_score
                        self.best_data = copy.deepcopy(self.process_data)
                        self.best_binary_op = tmp_op_list
                        print(f"best_score:{best_score}")
                        XX = self.best_data.copy()
                        XX['label'] = self.target
                        XX.to_csv('data_log/' + self.args.data + '_' + "ppo" + str(best_score) + '_output.csv',
                                  index=False)

            self.best_data = self.new_data_pipeline.old_pipline_main(self.best_data)
            self.best_data = pd.DataFrame(self.best_data)
            self.process_data = self.best_data
            self.cluster_dict = cluster_features(self.process_data, self.target)
            self.controller = Controller(self.args, self.cluster_dict, self.best_data, self.ori_data).to(self.device)
            for try_num in trange(self.episodes, desc='Episodes'):
                tmp = []
                sample_history = self.controller.sample(is_unary=True)
                unary_worker = unary_Worker(self.args, sample_history, is_uanry=True)
                tmp.append(unary_worker.actions_trans)
                tmp.append(try_num)
                ids_unary.append(tmp)
                unary_sample_workers.append(unary_worker)
            for i in tqdm(ids_unary, desc='Unary IDs'):
                print(f"ids_unary:{i}")
                tmp_op_list = i[0]
                new_feature_dict, new_cluster_dict = mutation(self.cluster_dict, self.process_data, self.target,
                                                              tmp_op_list,
                                                              self.process_data.columns,
                                                              task_type=self.task_type)
                for j in range(len(new_feature_dict)):
                    self.process_data, self.cluster_dict = new_feature_dict[j], new_cluster_dict[j]
                    if len(self.process_data.columns) > self.max_len_features:
                        self.process_data = feature_selection(self.process_data, self.target, {'method': 'SelectBest',
                                                                                               'task_type': self.task_type},
                                                              self.cluster_dict,self.max_len_features)
                    score = self.get_reward_ins.downstream_task_new(self.process_data, self.target,
                                                                    task_type=self.task_type)
                    unary_sample_workers[i[1]].score_list.append(score)
                    print(f"score:{score},,,score----{best_score - score}")
                    if score > best_score:
                        best_score = score
                        self.base_ori_score = best_score
                        self.best_data = copy.deepcopy(self.process_data)
                        self.best_binary_op = tmp_op_list
                        print(f"best_score:{best_score}")
                        XX = self.best_data.copy()
                        XX['label'] = self.target
                        XX.to_csv('data_log/'+ self.args.data + '_'+"ppo"+ str(best_score) +'_output.csv', index=False)
                        # update baseline

            # update baseline
            sample_score = [worker.score_list for worker in unary_sample_workers]
            if self.baseline is None:
                self.baseline = np.mean(sample_score)
            else:
                for worker in unary_sample_workers:
                    self.baseline = self.baseline * self.baseline_weight + sum(worker.score_list) * (
                            1 - self.baseline_weight)

            score_lists = [worker.score_list for worker in unary_sample_workers]
            score_reward = [score_list - self.baseline for score_list in score_lists]
            score_reward_abs = [abs(reward) for reward in score_reward[0]]
            min_reward = min(score_reward_abs)
            max_reward = max(score_reward_abs)
            score_reward_minmax = []
            for reward in score_reward[0]:
                if reward < 0:
                    min_max_reward = -(abs(reward) - min_reward) / (max_reward - min_reward + 1e-6)
                elif reward >= 0:
                    min_max_reward = (abs(reward) - min_reward) / (max_reward - min_reward + 1e-6)
                else:
                    min_max_reward = None
                    print('error')
                score_reward_minmax.append(min_max_reward)
            score_reward_minmax = [i / 2 for i in score_reward_minmax]
            normal_reward = score_reward_minmax

            # PPO network update
            # Cumulative gradient update method
            for ppo_epoch in range(self.ppo_epochs):
                loss = 0
                for episode, worker in enumerate(unary_sample_workers):
                    prod_history = self.controller.get_prod(worker.history,is_unary=True)
                    loss += self.call_worker_loss(prod_history, worker.history, normal_reward)
                loss /= len(unary_sample_workers)
                logging.info(f'epoch: {ppo_epoch}, loss: {loss}')
                print(f'epoch: {ppo_epoch}, loss: {loss}')
                # # cal. loss
                loss.backward()
                self.adam = optim.Adam(params=self.controller.parameters(), lr=self.arch_lr)
                # update
                self.adam.step()
                # ->zero
                self.adam.zero_grad()



    def call_worker_loss(self, prod_history, sample_history, reward):
        """
        calulate loss
        """
        policy_loss_list = []
        prod_actions_p = []
        sample_actions_p = []
        actions_entropy = []
        num_reward = len(reward)
        reward = sum(reward) / num_reward
        for ph in prod_history['feature_engineering']:
            prod_actions_p.append(ph['prob_vec'])
            actions_entropy.append(ph['action_entropy'])

        for sh in sample_history['unary_feature_engineering']:
            for wh in sh[1]:
                sample_actions_p.append(wh['prob_vec'][0])

        actions_entropy = torch.cat(actions_entropy, dim=0)
        all_entropy = torch.sum(actions_entropy)
        prod_actions_p = torch.cat([tensor.unsqueeze(0) for tensor in prod_actions_p], dim=0)
        # sample_actions_p = torch.cat(sample_actions_p, dim=0)
        sample_actions_p = torch.cat([tensor.unsqueeze(0) for tensor in sample_actions_p], dim=0)
        action_importance = prod_actions_p.exp() / sample_actions_p.exp()

        clipped_action_importance = torch.clamp(action_importance, 1 - self.clip_epsion,1 + self.clip_epsion)
        action_reward = action_importance * reward
        clipped_action_reward = clipped_action_importance * reward
        action_reward = torch.min(action_reward, clipped_action_reward)
        # policy_loss_list.append(action_reward)
        # for index, action_p in enumerate(prod_actions_p):
        #     # action_importance = action_p / sample_actions_p[index][0]
        #     # clipped_action_importance = self.clip(action_importance)
        #     action_importance = actions_entropy.exp() / sample_actions_p.exp()
        #     clipped_action_importance = torch.clamp(action_importance, 1 - self.clip_epsion,
        #                                          1 + self.clip_epsion)
        #     action_reward = action_importance * reward
        #     clipped_action_reward = clipped_action_importance * reward
        #
        #     action_reward = torch.min(action_reward,clipped_action_reward)
        #     policy_loss_list.append(action_reward)

        if self.eval_method == 'mse':
            all_policy_loss = sum(action_reward)
        elif self.eval_method == 'f1_score' or self.eval_method == 'r_squared' or self.eval_method == 'acc' or \
                self.eval_method == 'auc' or self.eval_method == 'ks' or self.eval_method == 'sub_rae':
            all_policy_loss = -1 * sum(action_reward)
        else:
            logging.info(f'incre defaut')
            all_policy_loss = -1 * sum(action_reward)

        entropy_bonus = -1 * all_entropy * self.entropy_weight
        return all_policy_loss + entropy_bonus

    def clip(self, actions_importance):
        lower = (torch.ones_like(actions_importance) * (1 - self.clip_epsilon))
        upper = (torch.ones_like(actions_importance) * (1 + self.clip_epsilon))
        # x = torch.cat([actions_importance, upper],dim=0)
        actions_importance = torch.min(lower,upper)
        actions_importance = torch.max(actions_importance, lower)


        return actions_importance

    def get_final_data(self, mas, new_data):
        """
        mas: 分数集
        new_data：数据集
        得到 分数对应最大的数据集
        """
        i = mas.index(max(mas))
        return new_data[i]



