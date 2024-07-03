from .self_attention_rnn import Actor
import torch
import torch.optim as optim
import os
from torch.distributions.categorical import Categorical
import logging


class PPO(object):
    def __init__(self, args, data_nums, operations_c, operations_d, d_model, d_k, d_v, d_ff, n_heads,
                 device, dropout=None, c_param=False):
        self.args = args
        self.entropy_weight = args.entropy_weight

        self.epochs = args.epochs
        self.episodes = args.episodes
        self.ppo_epochs = args.ppo_epochs
        self.operations_c = operations_c
        self.operations_d = operations_d
        self.device = device

        self.actor = Actor(args, data_nums, operations_c, operations_d, d_model, d_k, d_v, d_ff, n_heads,
                           rnn_hidden_size=128, dropout=dropout).to(self.device)
        self.actor_opt = optim.Adam(params=self.actor.parameters(), lr=args.lr)

        self.baseline = {}
        for step in range(args.steps_num):
            self.baseline[step] = None

        self.baseline_weight = self.args.baseline_weight

        self.clip_epsion = 0.2

    def choose_action(self, input_c, step, epoch, ops,res_h_c_t_list):
        actions_ops = []
        log_ops_probs = []
        actions_otp = []
        log_otp_probs = []

        self.actor.train()
        ops_softmax, otp_softmax,res_h_c_t_list = self.actor(input_c.to(self.device), step,res_h_c_t_list)

        for index, out in enumerate(ops_softmax):
            dist = Categorical(out)
            action_ops = dist.sample()
            log_prob = dist.log_prob(action_ops)
            actions_ops.append(int(action_ops.item()))
            log_ops_probs.append(log_prob.item())

        for index, out in enumerate(otp_softmax):
            dist = Categorical(out)
            action_otp = dist.sample()
            log_prob = dist.log_prob(action_otp)
            actions_otp.append(int(action_otp.item()))
            log_otp_probs.append(log_prob.item())
        return actions_ops, log_ops_probs, actions_otp, log_otp_probs,res_h_c_t_list

    def update(self, workers):
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
            for i, x in enumerate(worker.log_otp_probs):
                for j, item in enumerate(x):
                    workers[worker_index].log_otp_probs[i][j] = torch.tensor(item, device=self.device)

            for i, x in enumerate(worker.actions_ops):
                for j, item in enumerate(x):
                    workers[worker_index].actions_ops[i][j] = torch.tensor(item, device=self.device)
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
                for (otp_list,ops_list) in (zip(worker.log_otp_probs,worker.log_ops_probs)):
                    old_log_probs_.append(otp_list + ops_list)

                states = worker.states
                actions_otp = worker.actions_otp
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
                    action_otp = actions_otp[index]
                    action_ops = actions_ops[index]
                    step = steps[index]
                    if index == 0:
                        res_h_c_t_list = None
                    ops_softmax, otp_softmax,res_h_c_t_list = self.actor(state.to(self.device), step,res_h_c_t_list)
                    for k, out in enumerate(ops_softmax):
                        dist = Categorical(out)
                        entropy = dist.entropy()
                        entropys.append(entropy.unsqueeze(dim=0))
                        new_log_ops_prob = dist.log_prob(action_ops[k]).unsqueeze(dim=0).float()
                        new_log_probs.append(new_log_ops_prob)
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
            self.actor_opt.zero_grad()
            total_loss.backward()
            self.actor_opt.step()
