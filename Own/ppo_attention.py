import argparse
import copy

import pandas as pd
from tqdm import trange
from tqdm import tqdm
from Own.feature_eng.feature_selection import feature_selection
from Own.feature_eng.feature_cluster import cluster_features, cluster_features_1
import torch.optim as optim
from worker import unary_Worker, binary_Worker
import numpy as np
import torch
import logging
from reward import GetReward
from Own.Evolutionary_FE.Crossover import crossover
from Own.Evolutionary_FE.Mutation import mutation
from Own.feature_eng.data_trans import Pipeline_data
from Own.controller_lstm_attention import Controller
from Own.Evolutionary_FE.DNA_Fitness import fitness_score


class PPO_psm(object):
    def __init__(self, args):
        self.base_ori_score = None
        self.clip_epsion = 0.2
        self.num_step = 3
        self.args = args
        self.arch_lr = 0.001
        self.episodes = 5
        self.entropy_weight = 1e-3
        self.policy_nums = 30
        self.task_type = args.task_type
        self.eval_method = args.eval_method
        self.final_action_ori = None
        self.ppo_epochs = 20
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
        self.f_ori_data = self.ori_data.iloc[:, :-1]
        self.target = self.ori_data.iloc[:, -1]
        self.new_data = None
        self.max_len_features = 2000
        self.binary = None
        self.best_binary_op = None
        self.best_unary_op = None
        self.new_data_pipeline = Pipeline_data(all_data=self.ori_data, continuous_columns=None, discrete_columns=None)

    def get_ori_base_score(self):
        score_list = self.get_reward_ins.downstream_task_new(self.f_ori_data, self.target, task_type=self.task_type)
        return score_list

    def feature_search(self):
        self.base_ori_score = self.get_ori_base_score()
        print(f"thrdfd:{self.base_ori_score}")
        self.new_data = []


        #   根据新生成的process_data进行多轮采样一元操作集,并进行PPO
        for k in range(self.policy_nums):
            print(f"policy_nums:{k}_unary")
            best_score = self.base_ori_score
            unary_workers = []

            ma = -1 * float("inf")
            self.process_data = copy.deepcopy(self.ori_data)
            self.target = self.process_data.iloc[:, -1]
            self.process_data = self.process_data.iloc[:, :-1]
            if k == 0:
                self.best_data = copy.deepcopy(self.process_data)

            # print(f"policy_nums:{k}_unary")
            # best_score = self.base_ori_score
            # unary_workers = []
            # ma = -1 * float("inf")
            # self.best_data = self.new_data_pipeline.old_pipline_main(self.best_data)
            # self.best_data = pd.DataFrame(self.best_data)
            # self.process_data = self.best_data.copy()
            self.cluster_dict = cluster_features(self.process_data, self.target)
            self.controller = Controller(self.args, self.cluster_dict, self.process_data, self.ori_data).to(self.device)
            print(self.process_data.shape)
            for try_num in trange(self.episodes, desc='Episodes'):
                sample_unary_worker = self.controller.sample(is_unary=True, args=self.args)
                unary_workers.append(sample_unary_worker)
            for unary_worker in tqdm(unary_workers, desc='Unary IDs'):
                for i in range(self.num_step):
                    new_df = unary_worker.states_u[i]
                    score = fitness_score(new_df, self.target, self.task_type)
                    unary_worker.score_list.append(score)
                    print(f"score_unary:{score}_{i}")
                    if score > best_score:
                        best_score = score
                        self.base_ori_score = best_score
                        self.best_data = copy.deepcopy(new_df)
                        print(f"best_score:{best_score}")
                        XX = self.best_data.copy()
                        XX['label'] = self.target
                        XX.to_csv('output.csv', index=False)

            # update baseline
            sample_score = [worker.score_list for worker in unary_workers]
            if self.baseline is None:
                self.baseline = np.mean(sample_score)
            else:
                for worker in unary_workers:
                    self.baseline = self.baseline * self.baseline_weight + sum(worker.score_list) * (
                            1 - self.baseline_weight)

            score_lists = [worker.score_list for worker in unary_workers]
            score_reward = [score_list - self.baseline for score_list in score_lists]
            score_reward_abs = [sum(abs(value) for value in reward) for reward in score_reward]
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
                for episode, unary_worker in enumerate(unary_workers):
                    prob_unary_worker = self.controller.unary_new_prob(unary_worker)
                    loss += self.call_worker_loss(prob_unary_worker, unary_worker, normal_reward, is_unary=True)
                loss /= len(unary_workers)
                logging.info(f'epoch: {ppo_epoch}, loss: {loss}')
                # # cal. loss
                loss.backward()
                self.adam = optim.Adam(params=self.controller.parameters(), lr=self.arch_lr)
                # update
                self.adam.step()
                # ->zero
                self.adam.zero_grad()


        #   根据新生成的process_data进行多轮采样二元操作集
        for k in range(self.policy_nums):
            print(f"policy_nums:{k}_binary")
            best_score = self.base_ori_score
            binary_workers = []
            ma = -1 * float("inf")
            self.best_data = self.new_data_pipeline.old_pipline_main(self.best_data)
            self.best_data = pd.DataFrame(self.best_data)
            self.best_data.columns = self.best_data.columns.astype(str)
            # self.process_data = pd.concat([self.best_data, self.ori_data.iloc[:,:-1]], axis=1)
            self.process_data = self.best_data
            # print(f"policy_nums:{k}_binary")
            # best_score = self.base_ori_score
            # binary_workers = []
            #
            # ma = -1 * float("inf")
            # self.process_data = copy.deepcopy(self.ori_data)
            # self.target = self.process_data.iloc[:, -1]
            # self.process_data = self.process_data.iloc[:, :-1]
            # if k == 0:
            #     self.best_data = copy.deepcopy(self.process_data)
            self.cluster_dict = cluster_features(self.process_data, self.target)
            self.controller = Controller(self.args, self.cluster_dict, self.process_data, self.ori_data).to(self.device)
            for try_num in trange(self.episodes, desc='Episodes'):
                sample_binary_worker = self.controller.sample(is_unary=False, args=self.args)
                binary_workers.append(sample_binary_worker)
            for binary_worker in tqdm(binary_workers, desc='Binary IDs'):
                for i in range(self.num_step):
                    new_df = binary_worker.states_b[i]
                    print(new_df.shape)
                    score = fitness_score(new_df, self.target, self.task_type)
                    print(f"score_binary:{score}_{i}")
                    binary_worker.score_list.append(score)
                    if score > best_score:
                        best_score = score
                        self.base_ori_score = best_score
                        self.best_data = new_df.copy()
                        print(f"best_score:{best_score}")
                        XX = self.best_data.copy()
                        XX['label'] = self.target
                        XX.to_csv('output.csv', index=False)

            # update baseline
            sample_score = [worker.score_list for worker in binary_workers]
            if self.baseline is None:
                self.baseline = np.mean(sample_score)
            else:
                for worker in binary_workers:
                    self.baseline = self.baseline * self.baseline_weight + sum(worker.score_list) * (
                            1 - self.baseline_weight)

            score_lists = [worker.score_list for worker in binary_workers]
            score_reward = [score_list - self.baseline for score_list in score_lists]
            score_reward_abs = [sum(abs(value) for value in reward) for reward in score_reward]
            min_reward = min(score_reward_abs)
            max_reward = max(score_reward_abs)
            score_reward_minmax = []
            for reward in score_reward:
                avr_reward = np.mean(reward)
                if avr_reward < 0:
                    min_max_reward = -(abs(avr_reward) - min_reward) / (max_reward - min_reward + 1e-6)
                elif avr_reward >= 0:
                    min_max_reward = (abs(avr_reward) - min_reward) / (max_reward - min_reward + 1e-6)
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
                for episode, binary_worker in enumerate(binary_workers):
                    prob_binary_worker = self.controller.binary_new_prob(binary_worker)
                    loss += self.call_worker_loss(prob_binary_worker, binary_worker, normal_reward, is_unary=False)
                loss /= len(binary_workers)
                logging.info(f'epoch: {ppo_epoch}, loss: {loss}')
                # # cal. loss
                loss.backward()
                self.adam = optim.Adam(params=self.controller.parameters(), lr=self.arch_lr)
                # update
                self.adam.step()
                # ->zero
                self.adam.zero_grad()

    def call_worker_loss(self, prob_worker, worker, reward, is_unary):
        """
        calulate loss
        """

        num_reward = len(reward)
        reward = sum(reward) / num_reward
        policy_loss_list = []
        if is_unary:
            prod_actions_p = prob_worker.prob_vecs_u
            actions_entropy = prob_worker.action_entropys_u
            sample_actions_p = worker.prob_vecs_u
        else:
            prod_actions_p = prob_worker.prob_vecs_b
            actions_entropy = prob_worker.action_entropys_b
            sample_actions_p = worker.prob_vecs_b

        actions_entropy = torch.cat(actions_entropy, dim=0)
        all_entropy = torch.sum(actions_entropy)

        for index, action_p in enumerate(prod_actions_p):
            action_importance = action_p.exp() / sample_actions_p[index][0].exp()

            clipped_action_importance = torch.clamp(action_importance, 1 - self.clip_epsion, 1 + self.clip_epsion)
            action_reward = action_importance * reward
            clipped_action_reward = clipped_action_importance * reward
            action_reward = torch.min(action_reward, clipped_action_reward)
            policy_loss_list.append(torch.sum(action_reward))

        if self.eval_method == 'mse':
            all_policy_loss = sum(policy_loss_list)
        elif self.eval_method == 'f1_score' or self.eval_method == 'r_squared' or self.eval_method == 'acc' or \
                self.eval_method == 'auc' or self.eval_method == 'ks' or self.eval_method == 'sub_rae':
            all_policy_loss = -1 * sum(policy_loss_list)
        else:
            logging.info(f'incre defaut')
            all_policy_loss = -1 * sum(policy_loss_list)

        entropy_bonus = -1 * all_entropy * self.entropy_weight
        return all_policy_loss + entropy_bonus

    def clip(self, actions_importance):
        lower = (torch.ones_like(actions_importance) * (1 - self.clip_epsilon))
        upper = (torch.ones_like(actions_importance) * (1 + self.clip_epsilon))
        actions_importance = torch.min(lower, upper)
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



