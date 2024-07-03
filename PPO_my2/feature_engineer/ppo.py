from .self_attention_rnn import Actor1, Actor2
import torch
import torch.optim as optim
import os
from torch.distributions.categorical import Categorical
import logging


class PPO(object):
    def __init__(self, args, data_nums, feature_nums, operations_c, operations_d, d_model, hidden_size, d_k, d_v, d_ff,
                 n_heads,
                 device, dropout=None, c_param=False):
        self.args = args
        self.entropy_weight = args.entropy_weight

        self.epochs = args.epochs
        self.episodes = args.episodes
        self.ppo_epochs = args.ppo_epochs
        self.operations_c = operations_c
        self.operations_d = operations_d
        self.device = device

        self.agent1 = Actor1(args, data_nums, feature_nums, operations_c, d_model, d_k, d_v, d_ff, n_heads,
                           self.device,dropout=dropout).to(self.device)
        self.agent1_opt = optim.Adam(params=self.agent1.parameters(), lr=args.lr)
        self.agent2 = Actor2(args, operations_c, hidden_size, d_model,data_nums).to(self.device)
        self.agent2_opt = optim.Adam(params=self.agent2.parameters(), lr=args.lr)

        self.baseline = {}
        for step in range(args.steps_num):
            self.baseline[step] = None

        self.baseline_weight = self.args.baseline_weight

        self.clip_epsion = 0.2

    def choose_action_ops(self, input_c, step, epoch, ops,con_or_dis):
        actions_ops = []
        log_ops_probs = []
        self.agent1.train()
        ops_softmax, state = self.agent1(input_c.to(self.device),con_or_dis)

        for index, out in enumerate(ops_softmax):
            dist = Categorical(out)
            action_ops = dist.sample()
            log_prob = dist.log_prob(action_ops)
            actions_ops.append(int(action_ops.item()))
            log_ops_probs.append(log_prob.item())

        return actions_ops, log_ops_probs, state

    def choose_action_otp(self, input_c,emb_state):
        actions_otp = []
        log_otp_probs = []
        self.agent2.train()
        input_c = torch.tensor(input_c)
        otp_softmax, state = self.agent2(input_c.to(self.device), emb_state)

        for index, out in enumerate(otp_softmax):
            dist = Categorical(out)
            action_otp = dist.sample()
            log_prob = dist.log_prob(action_otp)
            actions_otp.append(int(action_otp.item()))
            log_otp_probs.append(log_prob.item())

        return actions_otp, log_otp_probs

    def update_ops(self, workers):
        rewards = []
        dones = []
        for worker in workers:
            rewards.extend(worker.accs)
            dones.extend(worker.dones)

        rewards_convert = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gama * discounted_reward)
            rewards_convert.insert(0, discounted_reward)
        for step in range(self.args.steps_num):
            for reward in rewards_convert[step::self.args.steps_num]:
                if self.baseline[step] == None:
                    self.baseline[step] = reward
                else:
                    self.baseline[step] = self.baseline[step] * self.baseline_weight + reward * (
                            1 - self.baseline_weight)
        baseline_step = []
        for step in range(self.args.steps_num):
            baseline_step.append(self.baseline[step])

        baseline_step = torch.tensor(baseline_step, device=self.device)
        self.baseline_step = baseline_step
        rewards_convert = torch.tensor(rewards_convert, device=self.device).reshape(-1, self.args.steps_num)
        advantages = rewards_convert - baseline_step

        # Move tensor in worker to self.device
        for worker_index, worker in enumerate(workers):
            for i, x in enumerate(worker.log_ops_probs):
                for j, item in enumerate(x):
                    workers[worker_index].log_ops_probs[i][j] = torch.tensor(item, device=self.device)

            for i, x in enumerate(worker.actions_ops):
                for j, item in enumerate(x):
                    workers[worker_index].actions_ops[i][j] = torch.tensor(item, device=self.device)
            for index, state in enumerate(worker.states):
                workers[worker_index].states[index] = state.to(self.device)

        for epoch in range(self.args.ppo_epochs):
            total_loss = 0
            total_loss_actor = 0
            total_loss_entorpy = 0

            for worker_index, worker in enumerate(workers):
                old_log_probs_ = []
                for ops_list in worker.log_ops_probs:
                    old_log_probs_.append(ops_list)

                states = worker.states
                actions_ops = worker.actions_ops
                steps = worker.steps

                advantage = advantages[worker_index]
                advantage_convert = []

                for i, log_pros in enumerate(old_log_probs_):
                    advantage_ = advantage[i]
                    for j, log_pro in enumerate(log_pros):
                        advantage_convert.append(advantage_)
                advantage_convert = torch.tensor(advantage_convert, device=self.device)

                old_log_probs = torch.tensor([item for x in old_log_probs_ for item in x], device=self.device)

                new_log_probs = []
                entropys = []
                for index, state in enumerate(states):
                    action_ops = actions_ops[index]
                    ops_softmax, emb_state = self.agent1(state.to(self.device),worker.con_or_diss[index])
                    for k, out in enumerate(ops_softmax):
                        dist = Categorical(out)
                        entropy = dist.entropy()
                        entropys.append(entropy.unsqueeze(dim=0))
                        new_log_ops_prob = dist.log_prob(action_ops[k]).unsqueeze(dim=0).float()
                        new_log_probs.append(new_log_ops_prob)

                new_log_probs = torch.cat(new_log_probs)
                entropys = torch.cat(entropys)

                # ppo
                prob_ratio = new_log_probs.exp() / old_log_probs.exp()
                weighted_probs = advantage_convert * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_epsion,
                                                     1 + self.clip_epsion) * advantage_convert
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                actor_loss = actor_loss.sum()

                entropy_loss = entropys.sum()
                total_loss_actor += actor_loss
                total_loss_entorpy += (- self.args.entropy_weight * entropy_loss)
                total_loss += (actor_loss - self.args.entropy_weight * entropy_loss)
            factor = len(workers)
            total_loss /= factor
            actor_loss = total_loss_actor / factor
            entropy_loss = total_loss_entorpy / factor
            logging.info(
                f"total_loss_c:{total_loss.item()},actor_loss:{actor_loss.item()},entory_loss:{entropy_loss.item()}")
            self.agent1_opt.zero_grad()
            total_loss.backward()
            self.agent1_opt.step()

    def update_otp(self, workers):
        rewards = []
        dones = []
        for worker in workers:
            rewards.extend(worker.accs)
            dones.extend(worker.dones)

        rewards_convert = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gama * discounted_reward)
            rewards_convert.insert(0, discounted_reward)
        for step in range(self.args.steps_num):
            for reward in rewards_convert[step::self.args.steps_num]:
                if self.baseline[step] == None:
                    self.baseline[step] = reward
                else:
                    self.baseline[step] = self.baseline[step] * self.baseline_weight + reward * (
                            1 - self.baseline_weight)
        baseline_step = []
        for step in range(self.args.steps_num):
            baseline_step.append(self.baseline[step])

        baseline_step = torch.tensor(baseline_step, device=self.device)
        self.baseline_step = baseline_step
        rewards_convert = torch.tensor(rewards_convert, device=self.device).reshape(-1, self.args.steps_num)
        advantages = rewards_convert - baseline_step

        # Move tensor in worker to self.device
        for worker_index, worker in enumerate(workers):
            for i, x in enumerate(worker.log_otp_probs):
                for j, item in enumerate(x):
                    workers[worker_index].log_otp_probs[i][j] = torch.tensor(item, device=self.device)

            for i, x in enumerate(worker.actions_otp):
                for j, item in enumerate(x):
                    workers[worker_index].actions_otp[i][j] = torch.tensor(item, device=self.device)
            for index, state in enumerate(worker.states):
                workers[worker_index].states[index] = state.to(self.device)

        for epoch in range(self.args.ppo_epochs):
            total_loss = 0
            total_loss_actor = 0
            total_loss_entorpy = 0

            for worker_index, worker in enumerate(workers):
                old_log_probs_ = []
                for otp_list in worker.log_otp_probs:
                    old_log_probs_.append(otp_list)

                emb_states = worker.emb_states
                actions_otp = worker.actions_otp
                states2 = worker.states2


                advantage = advantages[worker_index]
                advantage_convert = []

                for i, log_pros in enumerate(old_log_probs_):
                    advantage_ = advantage[i]
                    for j, log_pro in enumerate(log_pros):
                        advantage_convert.append(advantage_)
                advantage_convert = torch.tensor(advantage_convert, device=self.device)

                old_log_probs = torch.tensor([item for x in old_log_probs_ for item in x], device=self.device)

                new_log_probs = []
                entropys = []
                for index, (state2,emb_state) in enumerate(zip(states2,emb_states)):
                    # action_ops = torch.tensor([item for item in states2]).to(self.device)
                    state2 = state2.to(self.device)
                    otp_softmax,all_embedding = self.agent2(state2,emb_state.to(self.device))
                    action_otp = actions_otp[index]
                    for k, out in enumerate(otp_softmax):
                        dist = Categorical(out)
                        entropy = dist.entropy()
                        entropys.append(entropy.unsqueeze(dim=0))
                        new_log_otp_prob = dist.log_prob(action_otp[k]).unsqueeze(dim=0).float()
                        new_log_probs.append(new_log_otp_prob)

                new_log_probs = torch.cat(new_log_probs)
                entropys = torch.cat(entropys)

                # ppo
                prob_ratio = new_log_probs.exp() / old_log_probs.exp()
                weighted_probs = advantage_convert * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_epsion,
                                                     1 + self.clip_epsion) * advantage_convert
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                actor_loss = actor_loss.sum()

                entropy_loss = entropys.sum()
                total_loss_actor += actor_loss
                total_loss_entorpy += (- self.args.entropy_weight * entropy_loss)
                total_loss += (actor_loss - self.args.entropy_weight * entropy_loss)
            factor = len(workers)
            total_loss /= factor
            actor_loss = total_loss_actor / factor
            entropy_loss = total_loss_entorpy / factor
            logging.info(
                f"total_loss_c:{total_loss.item()},actor_loss:{actor_loss.item()},entory_loss:{entropy_loss.item()}")
            self.agent2_opt.zero_grad()
            total_loss.backward()
            self.agent2_opt.step()
