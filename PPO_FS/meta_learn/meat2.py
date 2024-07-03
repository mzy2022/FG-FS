import torch
import torch.nn as nn
import torch.optim as optim
from meta_world.envs import ML1_REACH_V2
from tqdm import tqdm

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def maml_train(env, policy, inner_lr, outer_lr, num_iterations, num_tasks, num_shots, num_test_shots):
    optimizer = optim.Adam(policy.parameters(), lr=outer_lr)

    for iteration in tqdm(range(num_iterations)):
        meta_train_loss = 0
        meta_test_loss = 0

        for _ in range(num_tasks):
            # Sample a new task
            env.set_task(env.sample_tasks(1)[0])

            # Adapt the policy to the task
            adapted_policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
            adapted_policy.load_state_dict(policy.state_dict())
            adapted_optimizer = optim.Adam(adapted_policy.parameters(), lr=inner_lr)

            for _ in range(num_shots):
                obs, actions, rewards, next_obs, dones = env.sample_trajectory()
                loss = -adapted_policy(obs).gather(1, actions.long().unsqueeze(1)).mean()
                adapted_optimizer.zero_grad()
                loss.backward()
                adapted_optimizer.step()

            # Evaluate the adapted policy on the task
            obs, actions, rewards, next_obs, dones = env.sample_trajectory(num_test_shots)
            meta_train_loss += -adapted_policy(obs).gather(1, actions.long().unsqueeze(1)).mean()

            # Compute the meta-gradient and update the policy
            meta_train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return policy