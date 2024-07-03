import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Controller:
    def __init__(self, args, num_feature):
        self.num_feature = num_feature
        self.num_op_unary = args.num_op_unary
        self.num_op_binary = args.num_op_binary
        self.num_op = args.num_op_unary + (self.num_feature - 1) * args.num_op_binary + 1
        self.max_order = args.max_order
        self.num_batch = args.num_batch
        self.opt = args.optimizer
        self.lr = args.lr
        self.lr_value = args.lr_value
        self.num_action = self.num_feature * self.max_order
        self.reg = args.reg

    def _create_rnn(self):
        self.rnns = {}
        for i in range(self.num_feature):
            self.rnns[f'rnn{i}'] = nn.LSTMCell(
                input_size=self.num_op, hidden_size=self.num_op)

    def _create_placeholder(self):
        self.concat_action = torch.zeros((self.num_batch, self.num_action), dtype=torch.int64)
        self.rewards = torch.zeros((self.num_batch, self.num_action), dtype=torch.float32)
        self.state = torch.empty(self.num_action, dtype=torch.int32)
        self.value = torch.zeros(1, dtype=torch.float32)

    def _create_variable(self):
        self.input0 = torch.ones(
            size=[self.num_feature, self.num_op], dtype=torch.float32)
        self.input0 = self.input0 / self.num_op

        self.value_estimator = nn.Sequential(
            nn.Linear(self.num_action, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )
        self.value_optimizer = optim.Adam(self.value_estimator.parameters(), lr=self.lr_value)
        self.value_criterion = nn.MSELoss()

    def _create_inference(self):
        self.outputs = {}

        for i in range(self.num_feature):
            tmp_h = torch.zeros(1, self.num_op)
            tmp_c = torch.zeros(1, self.num_op)
            tmp_input = nn.Embedding(self.input0[i], 8)
            for order in range(self.max_order):
                tmp_input, (tmp_h, tmp_c) = self.rnns[f'rnn{i}'](tmp_input, (tmp_h, tmp_c))
                if order == 0:
                    self.outputs[f'output{i}'] = tmp_input
                else:
                    self.outputs[f'output{i}'] = torch.cat(
                        [self.outputs[f'output{i}'], tmp_input], dim=0)
        self.concat_output = torch.cat(list(self.outputs.values()), dim=0)

    def _create_loss(self):
        self.loss = 0.0
        for batch_count in range(self.num_batch):
            action = self.concat_action[batch_count]
            reward = self.rewards[batch_count]
            action_probs = torch.softmax(self.concat_output, dim=0).squeeze()
            pick_action_prob = action_probs.gather(0, action)
            loss_batch = torch.sum(-torch.log(pick_action_prob) * reward)
            loss_entropy = torch.sum(-action_probs * torch.log(action_probs)) * self.reg
            loss_reg = 0.0
            for i in range(self.num_feature):
                weights = [w for w in self.rnns[f'rnn{i}'].parameters()]
                for w in weights:
                    loss_reg += self.reg * torch.sum(torch.square(w))
            self.loss += loss_batch + loss_entropy + loss_reg

        self.loss /= self.num_batch

    def _create_optimizer(self):
        if self.opt == 'adam':
            self.optimizer = optim.Adam(params=self.parameters(), lr=self.lr)
        elif self.opt == 'adagrad':
            self.optimizer = optim.Adagrad(params=self.parameters(), lr=self.lr)

    @property
    def parameters(self):
        return list(self.rnns.parameters()) + list(self.value_estimator.parameters())

    def build_graph(self):
        self._create_rnn()
        self._create_variable()
        self._create_placeholder()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()

    def update_policy(self, feed_dict, optimizer):
        optimizer.zero_grad()
        loss = self.loss.backward()
        optimizer.step()
        return loss.item()

    def update_value(self, state, value_estimator, value_optimizer):
        value_optimizer.zero_grad()
        prediction = value_estimator(state)
        loss = self.value_criterion(prediction, value)
        loss.backward()
        value_optimizer.step()

    def predict_value(self, state, value_estimator):
        with torch.no_grad():
            value = value_estimator(state)
        return np.squeeze(value.numpy())

    def concat_output(self):
        return self.concat_output

class Controller_pure:
    def __init__(self, args, num_feature):
        self.num_feature = num_feature
        self.num_op_unary = args.num_op_unary
        self.num_op_binary = args.num_op_binary
        self.num_op = args.num_op_unary + (self.num_feature - 1) * args.num_op_binary + 1
        self.max_order = args.max_order
        self.num_batch = args.num_batch
        self.opt = args.optimizer
        self.lr = args.lr
        self.lr_value = args.lr_value
        self.num_action = self.num_feature * self.max_order
        self.reg = args.reg

    def concat_output(self):
        return self.concat_output

    def _create_placeholder(self):
        self.concat_action = torch.zeros((self.num_batch, self.num_action), dtype=torch.int64)
        self.rewards = torch.zeros((self.num_batch, self.num_action), dtype=torch.float32)
        self.state = torch.empty(self.num_action, dtype=torch.int32)
        self.value = torch.zeros(1, dtype=torch.float32)

    def _create_variable(self):
        self.input0 = np.ones(shape=[self.num_action, self.num_op], dtype=np.float32)
        self.input0 = self.input0 / self.num_op
        self.concat_output = torch.nn.Parameter(torch.Tensor(self.input0), requires_grad=True)

        self.value_estimator = nn.Sequential(
            nn.Linear(self.num_action, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )
        self.value_optimizer = optim.Adam(self.value_estimator.parameters(), lr=self.lr_value)
        self.value_criterion = nn.MSELoss()

    def _create_loss(self):
        self.loss = 0.0
        for batch_count in range(self.num_batch):
            action = self.concat_action[batch_count]
            reward = self.rewards[batch_count]

            action_index = torch.stack([torch.arange(self.num_action), action], dim=1)
            action_probs = torch.softmax(self.concat_output, dim=1).squeeze()
            pick_action_prob = action_probs[action_index]

            loss_batch = -torch.sum(reward * torch.log(torch.clamp(pick_action_prob, min=1e-10, max=1.0)))
            loss_entropy = -torch.sum(
                action_probs * torch.log(torch.clamp(action_probs, min=1e-10, max=1.0))) * self.reg

            self.loss += loss_batch + loss_entropy
        self.loss /= self.num_batch

    def _create_optimizer(self):
        if self.opt == 'adam':
            self.optimizer = optim.Adam([self.concat_output], lr=self.lr)
        elif self.opt == 'adagrad':
            self.optimizer = optim.Adagrad([self.concat_output], lr=self.lr)

    def build_graph(self):
        self._create_variable()
        self._create_placeholder()
        self._create_loss()
        self._create_optimizer()

    def update_policy(self, feed_dict, optimizer):
        optimizer.zero_grad()
        loss = self.loss.backward()
        optimizer.step()
        return loss.item()

    def update_value(self, state, value_estimator, value_optimizer):
        value_optimizer.zero_grad()
        prediction = value_estimator(state)
        loss = self.value_criterion(prediction, value)
        loss.backward()
        value_optimizer.step()

    def predict_value(self, state, value_estimator):
        with torch.no_grad():
            value = value_estimator(state)
        return np.squeeze(value.numpy())


class Controller_sequence(nn.Module):
    def __init__(self, args, num_feature):
        super(Controller_sequence, self).__init__()

        self.num_feature = num_feature
        self.num_op_unary = args.num_op_unary
        self.num_op_binary = args.num_op_binary
        self.num_op = args.num_op_unary + (self.num_feature - 1) * args.num_op_binary + 1
        self.max_order = args.max_order
        self.num_batch = args.num_batch
        self.lr = args.lr
        self.lr_value = args.lr_value
        self.num_action = self.num_feature * self.max_order

        self.rnn = nn.LSTM(self.num_op, self.num_op).double()
        input0_double = torch.ones((1, self.num_op)).double()
        self.input0 = input0_double / self.num_op

        self.value_estimator = nn.Sequential(
            nn.Linear(self.num_action, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, action=None, rewards=None):
        h = self.rnn.initHidden().double()

        for action_count in range(self.num_action):
            if action_count == 0:
                tmp_input = self.input0.view(1, 1, -1)
            out, h = self.rnn(tmp_input, h)
            if action_count == 0:
                outputs = out
            else:
                outputs = torch.cat((outputs, out), 0)

        concat_output = outputs.view(-1, self.num_op)
        action_probs = F.softmax(concat_output, dim=-1)

        loss = 0.0
        for batch_count in range(self.num_batch):
            log_prob = torch.log(action_probs)
            reward = rewards[batch_count]

            loss_batch = -log_prob.gather(1, action[batch_count].view(-1, 1)) * reward
            entropy = -action_probs * log_prob
            loss += loss_batch.sum() + entropy.sum()

        return loss / self.num_batch

    def predict_value(self, state):
        return self.value_estimator(state.float()).squeeze(1)

    def adjust_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @property
    def value_optimizer(self):
        return torch.optim.Adam(self.value_estimator.parameters(), lr=self.lr_value)
