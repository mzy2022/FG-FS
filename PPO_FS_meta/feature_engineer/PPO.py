import torch
import torch.optim as optim
import os
from torch.distributions.categorical import Categorical
import logging
from PPO_FS_meta.feature_engineer.embedding_policy_network import Policy
import numpy as np


class PPO(object):
    def __init__(self, args, data_nums,feature_nums,select_feature_nums, d_model, alpha,device):
        self.args = args
        self.entropy_weight = args.entropy_weight
        self.ppo_epochs = args.ppo_epochs
        self.meta_ppo_epochs = args.meta_ppo_epochs
        self.device = device

        self.policy = Policy(args, data_nums, feature_nums, select_feature_nums,d_model, alpha, self.device).to(self.device)
        self.policy_opt = optim.Adam(params=self.policy.parameters(), lr=args.lr)
        x = self.policy.parameters()
        self.baseline = None
        self.baseline_weight = args.baseline_weight
        self.clip_epsion = 0.2

    def choose_action(self, input_x, input_y, feature_nums, adj_maxtrix):
        self.policy.train()
        action_softmax = self.policy(input_x.to(self.device), input_y.to(self.device),adj_maxtrix)
        action = torch.multinomial(action_softmax, num_samples=feature_nums, replacement=False)
        dist = Categorical(action_softmax)
        log_prob = dist.log_prob(action)
        return action, log_prob, action_softmax

    def meta_update(self, workers):
        rewards = []
        for worker in workers:
            reward = worker.reward_
            rewards.append(reward)
        for reward in rewards:
            if self.baseline is None:
                self.baseline = reward
            else:
                self.baseline = self.baseline * self.baseline_weight + reward * (1 - self.baseline_weight)
        self.baseline = torch.tensor(self.baseline, device=self.device)
        rewards = torch.tensor(rewards).to(self.device)
        advantages = rewards - self.baseline

        for epoch in range(self.meta_ppo_epochs):
            total_loss = 0
            total_loss_actor = 0
            total_loss_entorpy = 0
            for num,worker in enumerate(workers):
                old_log_probs = worker.inner_log_prob_.detach()
                input_x = worker.states_x.cpu()
                input_y = worker.states_y.cpu()
                adj_matrix = worker.adj_matrix
                actions = worker.inner_actions_

                new_log_probs = []
                action_softmax = self.policy(input_x.to(self.device),input_y.to(self.device),adj_matrix)
                dist = Categorical(action_softmax)
                entropy = dist.entropy()
                for action in actions:
                    new_log_prob = dist.log_prob(action).unsqueeze(dim=0).float()
                    new_log_probs.append(new_log_prob)
                new_log_probs = torch.cat(new_log_probs)


                # ppo
                prob_ratio = new_log_probs.exp() / old_log_probs.exp()
                weighted_probs = advantages[num] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_epsion,1 + self.clip_epsion) * advantages[num]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                actor_loss = actor_loss.sum()

                entropy_loss = entropy.sum()
                total_loss_actor += actor_loss
                total_loss_entorpy += (- self.args.entropy_weight * entropy_loss)
                total_loss += (actor_loss - self.args.entropy_weight * entropy_loss)

            total_loss /= len(workers)
            self.policy_opt.zero_grad()
            total_loss.backward()
            self.policy_opt.step()

    def update(self, worker):
        reward = worker.reward
        if self.baseline == None:
            self.baseline = reward
        else:
            self.baseline = self.baseline * self.baseline_weight + reward * (1 - self.baseline_weight)
        self.baseline = torch.tensor(self.baseline, device=self.device)
        advantage = reward - self.baseline

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
            action_softmax = self.policy(input_x.to(self.device),input_y.to(self.device),adj_matrix)
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
            total_loss += (actor_loss - self.args.entropy_weight * entropy_loss)

            self.policy_opt.zero_grad()
            total_loss.backward()
            self.policy_opt.step()