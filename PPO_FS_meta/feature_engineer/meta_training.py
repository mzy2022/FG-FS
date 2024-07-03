import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def inner_sample(train_data_x, train_data_y, feature_nums,adj_matrix,ppo, worker, device):
    # with torch.no_grad():
    init_x_state = torch.from_numpy(train_data_x.values).float().transpose(0, 1).to(device)
    init_y_state = torch.from_numpy(train_data_y.values).float().transpose(0, 1).to(device)
    actions, log_prob, action_softmax = ppo.choose_action(init_x_state, init_y_state,feature_nums,adj_matrix)
    worker.states_x = init_x_state
    worker.states_y = init_y_state
    worker.inner_actions = actions
    worker.inner_log_prob = log_prob
    worker.inner_action_softmax = action_softmax
    worker.adj_matrix = adj_matrix
    return worker



def inner_update(args,train_data_x, train_data_y, feature_nums,adj_matrix,ppo, worker, device):
    initial_params = [param.clone() for param in ppo.policy.parameters()]
    init_x_state = torch.from_numpy(train_data_x.values).float().transpose(0, 1).to(device)
    init_y_state = torch.from_numpy(train_data_y.values).float().transpose(0, 1).to(device)

    # with torch.no_grad():
    log_prob = worker.inner_log_prob
    reward = worker.inner_reward
    loss = - log_prob * reward
    loss = loss.sum() / len(log_prob)
    grads = torch.autograd.grad(loss, ppo.policy.parameters(), create_graph=True)
    fast_weights = [param - args.inner_lr * grad for param, grad in zip(ppo.policy.parameters(), grads)]
    # 更新worker的权重
    for param, fast_weight in zip(ppo.policy.parameters(), fast_weights):
        param.data.copy_(fast_weight.data)
    actions, log_prob, action_softmax = ppo.choose_action(init_x_state, init_y_state, feature_nums, adj_matrix)

    with torch.no_grad():
        for param, initial_param in zip(ppo.policy.parameters(), initial_params):
            param.data.copy_(initial_param.data)


    worker.fast_weight = fast_weights
    worker.inner_actions_ = actions
    worker.inner_log_prob_ = log_prob
    worker.inner_action_softmax_ = action_softmax
    return worker


def inner_get_reward(args, worker, X_train, Y_train, X_val, Y_val):
    actions = np.array(worker.inner_actions.cpu())
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X_train.iloc[:, actions], Y_train)
    y_pred = model.predict(X_val.iloc[:, actions])
    Y_val = Y_val.values
    accuracy = accuracy_score(y_pred, Y_val)
    worker.inner_reward = accuracy
    return worker,actions,accuracy

def outer_get_reward(args, worker, X_train, Y_train, X_val, Y_val):
    actions = np.array(worker.inner_actions_.cpu())
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X_train.iloc[:, actions], Y_train)
    y_pred = model.predict(X_val.iloc[:, actions])
    Y_val = Y_val.values
    accuracy = accuracy_score(y_pred, Y_val)
    worker.reward_ = accuracy
    return worker,actions,accuracy