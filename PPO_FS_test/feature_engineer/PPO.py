import torch
import torch.optim as optim
import os
from torch.distributions.categorical import Categorical
import logging
from PPO_FS_test.feature_engineer.embedding_policy_network import FNN,GAT
import numpy as np
import torch.nn.functional as F

class PPO(object):
    def __init__(self, args, data_nums,feature_nums,select_feature_nums, d_model, alpha,device):
        self.args = args
        self.entropy_weight = args.entropy_weight
        self.ppo_epochs = args.ppo_epochs

        self.device = device

        self.FNN = FNN(args, data_nums, feature_nums, select_feature_nums,d_model, alpha, self.device).to(self.device)
        self.GAT = GAT(args, data_nums, feature_nums, select_feature_nums, d_model, alpha, self.device).to(
            self.device)

        self.policy_opt = optim.Adam(set(list(self.FNN.parameters()) + list(self.GAT.parameters())), lr=args.lr)

        self.baseline = None
        self.baseline_weight = args.baseline_weight
        self.clip_epsion = 0.2

    def choose_action(self, input_x, input_y, feature_nums, adj_maxtrix):
        self.FNN.train()
        self.GAT.train()
        emb_x,emb_y = self.FNN(input_x.to(self.device), input_y.to(self.device), adj_maxtrix)
        action_softmax = self.GAT(emb_x.to(self.device), emb_y.to(self.device),adj_maxtrix)
        action = torch.multinomial(action_softmax, num_samples=feature_nums, replacement=False)
        dist = Categorical(action_softmax)
        log_prob = dist.log_prob(action)
        return action, log_prob, action_softmax,emb_x

    def feature_select(self,input_x,input_y,action,adj_matrix):
        matrix = torch.zeros((input_x.shape[0], input_x.shape[1])).to(self.device)
        matrix[:, action] = 1
        input_x = input_x * matrix

        with torch.no_grad():
            select_emb_x, emb_y = self.FNN(input_x.to(self.device), input_y.to(self.device),adj_matrix)
        return select_emb_x


    def update(self, worker):
        reward = worker.reward
        if self.baseline == None:
            self.baseline = reward
        else:
            self.baseline = self.baseline * self.baseline_weight + reward * (1 - self.baseline_weight)
        self.baseline = torch.tensor(self.baseline, device=self.device)
        advantage = reward - self.baseline
        worker.emb_x = worker.emb_x.detach()
        loss_mse = F.mse_loss(worker.emb_x, worker.select_emb_x)
        for epoch in range(self.ppo_epochs):
            total_loss = 0
            total_loss_actor = 0
            total_loss_entorpy = 0

            old_log_probs = worker.log_prob.detach()
            input_x = worker.states_x.cpu()
            input_y = worker.states_y.cpu()
            adj_matrix = worker.adj_matrix
            actions = worker.actions

            new_log_probs = []
            emb_x, emb_y = self.FNN(input_x.to(self.device), input_y.to(self.device), adj_matrix)
            action_softmax = self.GAT(emb_x.to(self.device), emb_y.to(self.device), adj_matrix)
            # action_softmax = self.policy(input_x.to(self.device),input_y.to(self.device),adj_matrix)

            dist = Categorical(action_softmax)
            entropy = dist.entropy()
            for action in actions:
                new_log_prob = dist.log_prob(action).unsqueeze(dim=0).float()
                new_log_probs.append(new_log_prob)
            new_log_probs = torch.cat(new_log_probs)


            # ppo
            prob_ratio = new_log_probs.exp() / old_log_probs.exp()
            weighted_probs = advantage * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_epsion,1 + self.clip_epsion) * advantage
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
            actor_loss = actor_loss.sum()

            entropy_loss = entropy.sum()
            total_loss_actor += actor_loss
            total_loss_entorpy += (- self.args.entropy_weight * entropy_loss)
            total_loss += (actor_loss - self.args.entropy_weight * entropy_loss + 0.01 * loss_mse)

            self.policy_opt.zero_grad()
            total_loss.backward()
            self.policy_opt.step()
