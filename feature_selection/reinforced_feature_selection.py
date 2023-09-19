import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

data_folder = 'D:/python files/pythonProject3/'
dataset = pd.read_csv(data_folder + 'wine_red.csv')

r, c = dataset.shape
array = dataset.values
X = dataset.iloc[:, 0:(c - 1)]
Y = dataset.iloc[:, (c - 1)]

X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=0.1, random_state=0)
Wilder_list = ['Wilderness_Area' + str(i) for i in range(1, 5)]
soil_list = ['Soil_Type' + str(i) for i in range(1, 41)]
binary_list = Wilder_list + soil_list

N_feature = X_train.shape[1]  # feature number
N_sample = X_train.shape[0]  # feature length,i.e., sample number

model = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=0)


def Feature_GCN(X):
    corr_matrix = X.corr().abs()
    corr_matrix[np.isnan(corr_matrix)] = 0
    corr_matrix_ = corr_matrix - np.eye(len(corr_matrix), k=0)
    sum_vec = corr_matrix_.sum()

    for i in range(len(corr_matrix_)):
        corr_matrix_.iloc[:, i] = corr_matrix_.iloc[:, i] / sum_vec[i]
        corr_matrix_.iloc[i, :] = corr_matrix_.iloc[i, :] / sum_vec[i]
    W = corr_matrix_ + np.eye(len(corr_matrix), k=0)
    Feature = np.mean(np.dot(X.values, W.values), axis=1)

    return Feature


#
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100  # After how much time you refresh target network
MEMORY_CAPACITY = 20  # The size of experience replay buffer
EXPLORE_STEPS = 30  # How many exploration steps you'd like, should be larger than MEMORY_CAPACITY
N_ACTIONS = 2
N_STATES = len(X_train)


#
#
class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQN(object):

    def __init__(self, N_STATES, N_ACTIONS):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY  # If full, restart from the beginning
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1])
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


np.random.seed(0)
action_list = np.random.randint(2, size=N_feature)

i = 0
while sum(action_list) < 2:
    np.random.seed(i)
    action_list = np.random.randint(2, size=N_feature)
    i += 1

X_selected = X_train.iloc[:, action_list == 1]
s = Feature_GCN(X_selected)

model.fit(X_train.iloc[:, action_list == 1], Y_train)
accuracy = model.score(X_val.iloc[:, action_list == 1], Y_val)
ave_corr = X_val.corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
r_list = (accuracy - 10 * ave_corr) / sum(action_list) * action_list

action_list_p = action_list


def QDN_main(s, action_list_p):
    dqn_list = []
    for agent in range(N_feature):
        dqn_list.append(DQN(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS))
    # The element in the result list consists two parts,
    # i.e., accuracy and the action list (action 1 means selecting corresponding feature, 0 means deselection).
    result = []

    for i in range(EXPLORE_STEPS):
        action_list = np.zeros(N_feature)
        for agent, dqn in enumerate(dqn_list):
            action_list[agent] = dqn.choose_action(s)

        while sum(action_list) < 2:
            np.random.seed(i)
            action_list = np.random.randint(2, size=N_feature)
            i += 1

        X_selected = X_train.iloc[:, action_list == 1]
        s_ = Feature_GCN(X_selected)

        model.fit(X_train.iloc[:, action_list == 1], Y_train)
        accuracy = model.score(X_val.iloc[:, action_list == 1], Y_val)
        ave_corr = X_val.corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])

        action_list_change = np.array([x or y for (x, y) in zip(action_list_p, action_list)])
        r_list = (accuracy - 10 * ave_corr) / sum(action_list_change) * action_list_change

        for agent, dqn in enumerate(dqn_list):
            dqn.store_transition(s, action_list[agent], r_list[agent], s_)

        if dqn_list[0].memory_counter > MEMORY_CAPACITY:
            for dqn in dqn_list:
                dqn.learn()
            print(sum(r_list), accuracy)
        s = s_
        action_list_p = action_list
        result.append([accuracy, action_list])
    return result


result = QDN_main(s, action_list_p)
print(result)

max_accuracy = 0
optimal_set = []
for i in range(len(result)):
    if result[i][0] > max_accuracy:
        max_accuracy = result[i][0]
        optimal_set = result[i][1]
print("The maximum accuracy is: {}, the optimal selection for each feature is:{}".format(max_accuracy, optimal_set))
