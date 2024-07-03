import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym


# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_action(self, state):
        logits = self.forward(state)
        action_probs = torch.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


# 元强化学习中的MAML算法实现
class MAML:
    def __init__(self, state_dim, action_dim, lr=0.01, meta_lr=0.001, K=3):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=meta_lr)
        self.lr = lr
        self.K = K

    def inner_update(self, states, actions, rewards):
        action_log_probs = []
        for state, action in zip(states, actions):
            _, log_prob = self.policy.get_action(state)
            action_log_probs.append(log_prob)
        loss = (-torch.stack(action_log_probs) * torch.tensor(rewards)).sum()
        grads = torch.autograd.grad(loss, self.policy.parameters())
        fast_weights = [param - self.lr * grad for param, grad in zip(self.policy.parameters(), grads)]
        return fast_weights

    def meta_update(self, tasks):
        meta_loss = 0.0
        for task in tasks:
            states, actions, rewards = task
            fast_weights = self.inner_update(states, actions, rewards)
            for _ in range(self.K - 1):
                fast_weights = self.inner_update(states, actions, rewards)
            meta_loss += self.compute_loss(states, actions, rewards, fast_weights)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

    def compute_loss(self, states, actions, rewards, weights):
        action_log_probs = []
        for state, action in zip(states, actions):
            logits = self.policy.forward(state)
            action_probs = torch.softmax(logits, dim=-1)
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(torch.tensor(action))
            action_log_probs.append(log_prob)
        loss = (-torch.stack(action_log_probs) * torch.tensor(rewards)).sum()
        return loss


# 环境和参数设置
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
maml = MAML(state_dim, action_dim)

# 训练过程
num_iterations = 1000
for iteration in range(num_iterations):
    reward = []
    tasks = []
    for _ in range(10):  # 生成10个任务
        states, actions, rewards = [], [], []
        state = env.reset()
        for _ in range(100):
            if _ == 0:
                state = state[0]
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, _ = maml.policy.get_action(state_tensor)
            xxx = env.step(action)
            next_state, reward, done, _,_ = xxx
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if done:
                break
        tasks.append((states, actions, rewards))
    maml.meta_update(tasks)
    print(sum(rewards))
    print(f"Iteration {iteration + 1}/{num_iterations} completed.")

env.close()

time.sleep(3)
env = gym.make('CartPole-v1',render_mode="human")
state = env.reset()
done = False
flag = 0
while not done:
    if flag == 0:
        state = state[0]
    flag += 1
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action, _ = maml.policy.get_action(state_tensor)
    next_state, reward, done, _,_ = env.step(action)
    env.render()
    state = next_state
env.close()
print(flag)

