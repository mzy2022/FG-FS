import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from tqdm import tqdm
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, eps_clip):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.policy(state)
        action = np.random.choice(len(action_probs.detach().numpy()[0]), p=action_probs.detach().numpy()[0])
        return action

    def update(self, memory):
        states, actions, rewards, next_states, dones = memory
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        old_probs, old_values = self.policy(states)
        old_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze(1).detach()

        _, new_values = self.policy(next_states)
        targets = rewards + self.gamma * new_values.squeeze(1) * (1 - dones)

        for _ in range(4):  # PPO update iterations
            probs, values = self.policy(states)
            probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratios = torch.exp(torch.log(probs) - torch.log(old_probs))
            advantages = targets - values.squeeze(1)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() + self.loss_fn(values.squeeze(1), targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class MAML_PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, eps_clip, inner_lr, meta_lr, num_inner_steps):
        self.policy = ActorCritic(state_dim, action_dim)
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=meta_lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.num_inner_steps = num_inner_steps
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.policy(state)
        action = np.random.choice(len(action_probs.detach().numpy()[0]), p=action_probs.detach().numpy()[0])
        return action

    def inner_update(self, memory):
        states, actions, rewards, next_states, dones = memory
        states = np.array(states)
        next_states = np.array(next_states)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        old_probs, old_values = self.policy(states)
        old_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze(1).detach()

        _, new_values = self.policy(next_states)
        targets = rewards + self.gamma * new_values.squeeze(1) * (1 - dones)

        fast_weights = list(self.policy.parameters())

        for _ in range(self.num_inner_steps):
            probs, values = self.policy(states)
            probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratios = torch.exp(torch.log(probs) - torch.log(old_probs))
            advantages = targets - values.squeeze(1)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() + self.loss_fn(values.squeeze(1), targets)

            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]

        return fast_weights

    def meta_update(self, tasks_memory):
        meta_loss = 0

        for memory in tasks_memory:
            fast_weights = self.inner_update(memory)

            states, actions, rewards, next_states, dones = memory
            states = np.array(states)
            next_states = np.array(next_states)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            old_probs, old_values = self.policy(states)
            old_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze(1).detach()

            _, new_values = self.policy(next_states)
            targets = rewards + self.gamma * new_values.squeeze(1) * (1 - dones)

            probs, values = self.policy(states)
            probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratios = torch.exp(torch.log(probs) - torch.log(old_probs))
            advantages = targets - values.squeeze(1)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() + self.loss_fn(values.squeeze(1), targets)
            meta_loss += loss

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

def collect_trajectory(env, policy, max_steps=200):
    state = env.reset()[0]
    states, actions, rewards, next_states, dones = [], [], [], [], []

    for _ in range(max_steps):
        action = policy.select_action(state)
        next_state, reward, done, _ ,_= env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        state = next_state
        if done:
            break

    return states, actions, rewards, next_states, dones

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

maml_ppo = MAML_PPO(state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, inner_lr=0.01, meta_lr=0.001, num_inner_steps=1)

for meta_iter in tqdm(range(1000)):
    tasks_memory = [collect_trajectory(env, maml_ppo) for _ in range(10)]
    maml_ppo.meta_update(tasks_memory)
