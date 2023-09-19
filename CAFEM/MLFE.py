import torch
import torch.nn as nn
import numpy as np
import random
import os
import math
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count, Process
import multiprocessing

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, f1_score, log_loss, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from scipy.io.arff import loadarff
from scipy import stats
import numpy.ma as ma
from sklearn.metrics import make_scorer

import os
from args import args

from utils import *


def load(f_path):
    '''
    这段代码定义了一个名为 load 的函数，用于加载数据集并进行预处理。
    :param f_path:
    :return:
    '''
    le = LabelEncoder()
    tasktype = ''
    if f_path[-4:] == 'arff':

        dataset, meta = loadarff(f_path)
        dataset = np.array(dataset.tolist())

        meta_names = meta.names()
        meta_types = meta.types()
        if meta_types[-1] == "nominal":
            tasktype = "C"
        else:
            tasktype = "R"
    for i, val in enumerate(meta_types):
        if val == "nominal":
            target = le.fit_transform(dataset[:, i]).astype(int)
            dataset[:, i] = target
    dataset = dataset.astype(float)
    return dataset, meta_types, tasktype


# Experience replay buffer
class Buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[max(1, int(0.0001 * self.buffer_size)):]

    def sample(self, size):
        if len(self.buffer) >= size:
            experience_buffer = self.buffer
        else:
            experience_buffer = self.buffer * size
        return np.copy(np.reshape(np.array(random.sample(experience_buffer, size)), [size, 5]))


def one_mse_func():
    def one_relative_abs(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        one_mae = 1 - mae / np.mean(np.abs(y_true - np.mean(y_true)))
        # print(one_mae,np.abs(one_mae))
        return np.abs(one_mae)


class Evaluater(object):
    def __init__(self, cv=5, stratified=True, n_jobs=1, tasktype="C", evaluatertype="rf", n_estimators=20,
                 random_state=np.random.randint(100000)):
        # evaluatertype = 'rf', 'svm', 'lr' for random forest, SVM, logisticregression
        # tasktype = "C" or "R" for classification or regression
        self.random_state = random_state
        self.cv = cv
        self.stratified = stratified
        self.n_jobs = n_jobs
        self.tasktype = tasktype
        if self.tasktype == "C":
            self.kf = StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        else:
            self.kf = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)

        if evaluatertype == 'rf':
            if tasktype == "C":
                self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state)
            elif tasktype == "R":
                self.clf = RandomForestRegressor(n_estimators=n_estimators, random_state=self.random_state)
        elif evaluatertype == "lr":
            if tasktype == "C":
                self.clf = LogisticRegression(solver='liblinear', random_state=self.random_state)
            elif tasktype == "R":
                self.clf = Lasso(random_state=self.random_state)

    def CV(self, X, y):
        X = np.nan_to_num(X)
        X = np.clip(X, -3e38, 3e38)
        scoring = 'f1' if self.tasktype == "C" else one_mse_func()
        score = cross_val_score(self.clf, X, y, scoring=scoring, cv=self.kf, n_jobs=self.n_jobs)
        return abs(score.mean())

    def CV2(self, X, y):
        res = []
        feature_importance = []
        for train_index, test_index in self.kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.clf.fit(X_train, y_train)
            y_test_hat = self.clf.predict(X_test)
            # feature_importance.append(self.clf.feature_importances_)
            res.append(self.metrics(y_test, y_test_hat))
        return np.array(res).mean(axis=0)

    def metrics(self, y_true, y_pred):
        if self.tasktype == "C":
            f_score = f1_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            logloss = log_loss(y_true, y_pred)
            return f_score, auc, logloss
        else:
            rel_MAE = 1 - mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true - np.mean(y_true)))
            rel_MSE = 1 - mean_squared_error(y_true, y_pred) / np.mean(np.square((y_true - np.mean(y_true))))
            return rel_MAE, rel_MSE


class ENV():
    def __init__(self, dataset, feature, globalreward=True, maxdepth=5, evalcount=10, binsize=100, opt_type='o1',
                 tasktype="C", evaluatertype='rf',
                 random_state=np.random.randint(100000), historysize=5, pretransform=None, n_jobs=1):
        if opt_type == 'o2':
            maxdepth = 1
        self.opt_type = opt_type
        self.historysize = historysize
        self.maxdepth = maxdepth
        self.globalreward = globalreward
        self.action = ['fs', 'square', 'tanh', 'round', 'log', 'sqrt', 'mmn', 'sigmoid', 'zscore'] \
            if opt_type == 'o1' else ['fs', 'sum', 'diff', 'product', 'divide']
        self.one_hot = np.array([0] * len(self.action))
        self.action_size = len(self.action)
        self.tasktype = tasktype
        self.evaluatertype = evaluatertype
        self.evalcount = evalcount
        self.random_state = random_state

        self.origin_dataset = dataset
        self.origin_feat = feature

        self._pretrf_mapper = [i for i in range(self.origin_dataset.shape[1])]
        if pretransform is not None:
            for act in pretransform:
                print(act)
                feat_id = act[0]
                actions = act[1].split("_")
                self.fe(actions, feat_id)
        if self.opt_type == 'o1':
            self.origin_feat = self._pretrf_mapper[self.origin_feat]
        elif self.opt_type == 'o2':
            value = []
            for val in self.origin_feat:
                value.append(self._pretrf_mapper[val])
            self.origin_feat = value

        self.evaluater = Evaluater(random_state=random_state, tasktype=tasktype, evaluatertype=evaluatertype,
                                   n_jobs=n_jobs)
        self._init_pfm = self.evaluater.CV(self.origin_dataset[:, :-1], self.origin_dataset[:, -1])
        self.init_pfm = self._init_pfm
        self.y = np.copy(self.origin_dataset[:, -1])
        self.binsize = binsize
        self._init()
        # print("init performance",self._init_pfm)

    def _init(self):
        self.dataset = np.copy(self.origin_dataset)
        self.feature = np.copy(self.dataset[:, self.origin_feat])
        self.transform = [0] * ((len(self.action) + 1) * self.historysize)
        self.now_pfm = self._init_pfm
        self.stop = False
        qsa_rep = self._QSA()
        self.state = np.concatenate([qsa_rep, np.copy(self.transform), [0] * len(self.action), [0] * len(self.action),
                                     [0] * len(self.action), [0] * len(self.action), [0, 0, 1, 0]], axis=None)
        self.tg = {0: {'p': self._init_pfm, 'd': 0}}
        self.nodeid = 0
        self.countnode = 1
        self.action_mask = np.array([0] * len(self.action))
        self.best_seq = []
        self.action_count = [0] * len(self.action)
        self.action_gain = [0.0] * len(self.action)
        self.node_visit = [0] * self.evalcount
        self.node_visit[0] = 1
        self.current_f = self.origin_feat
        self.tg[self.nodeid]['fid'] = self.current_f

    def node2root(self, adict, node):
        """
        函数通过迭代遍历从当前节点到根节点的路径，并将路径上的节点存储在列表 apath 中。
        :param adict:
        :param node:
        :return:
        """
        current_node = node
        apath = [node]
        while 'father' in adict[current_node]:
            current_node = adict[current_node]['father']
            apath.append(current_node)
        return [apath[i] for i in range(len(apath) - 1, -1, -1)][1:]

    def step(self, action):
        operator = self.action[action]
        if self.stop:
            return
        if operator == 'stop':
            self.stop = True
        elif operator == 'fs':
            if "father" in self.tg[self.nodeid]:
                self.nodeid = self.tg[self.nodeid]['father']
                performance = self.tg[self.nodeid]['p']
                self.current_f = self.tg[self.nodeid]['fid']
            else:
                self.tg[self.nodeid]['father'] = -1
                self.nodeid = -1
                performance = self.evaluater.CV(np.delete(self.origin_dataset, self.origin_feat, axis=1)[:, :-1],
                                                self.y)
                self.tg[self.nodeid] = {'p': performance, 'd': 1}
                self.stop = True

            reward = performance - self.now_pfm
            self.now_pfm = performance
        else:
            # feature was generated alreadly
            if operator in self.tg[self.nodeid]:
                newnode = self.tg[self.nodeid][operator]
                performance = self.tg[newnode]['p']
                self.nodeid = newnode
                self.current_f = self.tg[self.nodeid]['fid']
                reward = performance - self.now_pfm
                self.now_pfm = performance
            else:
                newfeature = feature = self.dataset[:, self.current_f]
                if self.opt_type == 'o1':
                    if operator in {'square', 'tanh', 'round'}:
                        newfeature = getattr(np, operator)(feature)
                    elif operator == "log":
                        vmin = feature.min()
                        newfeature = np.log(feature - vmin + 1) if vmin < 1 else np.log(feature)
                    elif operator == "sqrt":
                        vmin = feature.min()
                        newfeature = np.sqrt(feature - vmin) if vmin < 0 else np.sqrt(feature)
                    elif operator == "mmn":
                        mmn = MinMaxScaler()
                        newfeature = mmn.fit_transform(feature[:, np.newaxis]).flatten()
                    elif operator == "sigmoid":
                        newfeature = (1 + getattr(np, 'tanh')(feature / 2)) / 2
                    elif operator == 'zscore':
                        if np.var(feature) != 0:
                            ewfeature = stats.zscore(feature)
                elif self.opt_type == 'o2':
                    if operator == "sum":
                        newfeature = feature.sum(axis=1)
                    elif operator == "diff":
                        newfeature = feature[:, 0] * feature[:, 1]
                    elif operator == "product":
                        newfeature = feature[:, 0] * feature[:, 1]
                    elif operator == 'divide':
                        over = feature[:, 1]
                        while (np.any(over == 0)):
                            over = over + 1e-5
                        newfeature = feature[:, 0] / over

                if newfeature is not None:
                    newfeature = np.nan_to_num(newfeature)
                    newfeature = np.clip(newfeature, -math.sqrt(3.4e38), math.sqrt(3.4e38))
                    self.dataset = np.insert(self.dataset, self.dataset.shape[1] - 1, newfeature, axis=1)
                    self.current_f = self.dataset.shape[1] - 2
                else:
                    pass

                # X = np.concatenate([np.delete(self.dataset[:, :self.count_feat], self.origin_feat, 1), \
                #                    self.dataset[:, -2][:, np.newaxis]], axis=1)

                # apath = self.node2root(self.tg,self.nodeid)

                # X = np.concatenate([self.origin_dataset[:,:-1],\
                #                    self.dataset[:,[self.tg[v]['fid'] for v in apath]+[self.current_f]]],axis=1)
                X = np.concatenate([self.origin_dataset[:, :-1],
                                    self.dataset[:, [self.current_f]]], axis=1)
                # X = np.concatenate([np.delete(self.origin_dataset[:, :-1],self.origin_feat,axis=1), \
                #                    self.dataset[:, [self.current_f]]], axis=1)

                performance = self.evaluater.CV(X, self.y)
                self.tg[self.nodeid][operator] = self.countnode
                newnode = self.countnode
                self.tg[newnode] = {'p': performance, 'd': self.tg[self.nodeid]['d'] +1,'fid':self.current_f}
                self.tg[newnode]['father'] = self.nodeid
                self.countnode += 1
                self.nodeid = newnode
                if self.countnode >= self.evalcount:
                    self.stop = True
                reward = performance - self.now_pfm
                self.now_pfm = performance

        if self.stop:
            reward = 0

        if self.tg[self.nodeid]['d'] >= self.maxdepth:
            self.action_mask = [0] + [1] * (len(self.action) - 1)
        else:
            self.action_mask = [0] * len(self.action)

        if self.dataset.shape[1] <= 2:
            self.action_mask[0] = 1
        if self.countnode >= self.evalcount:
            self.action_mask = [1] * (len(self.action))

            # history seq
            onehot = np.copy(self.one_hot)
            onehot[action] = 1
            self.transform.extend(onehot)
            self.transform.append(reward)
            self.transform = self.transform[-self.historysize * (len(self.action) + 1):]
            # ExQSA
            qsa_rep = self._QSA()
            # action node visit
            self.node_visit[self.nodeid] += 1
            act_node_visit = [0] * len(self.action)
            for key in self.tg[self.nodeid]:
                if key in self.action:
                    act_node_visit[self.action.index(key)] = self.node_visit[self.tg[self.nodeid][key]]

            # count of action
            self.action_count[action] += 1
            if self.node_visit[self.nodeid] == 1:
                self.action_gain[action] += reward
            # gain each action
            gain_each = [0 if self.action_count[i] == 0 else self.action_gain[i] / self.action_count[i] for i in
                         range(len(self.action))]

            # count action from root
            action_count_root = [0] * len(self.action)
            startnode = self.nodeid
            while "father" in self.tg[startnode] and startnode > 0:
                lastnode = self.tg[startnode]['father']
                for key in self.tg[lastnode]:
                    if key in self.action and key != 'fid' and self.tg[lastnode][key] == startnode:
                        self.best_seq.insert(0, key)
                        action_count_root[self.action.index(key)] += 1
                        break
                startnode = lastnode
            if self.nodeid == -1:
                action_count_root[1] = 1
            # gain last and last last
            gain_last = reward
            gain_lastlast = 0
            if 'father' in self.tg[self.nodeid]:
                if 'father' in self.tg[self.tg[self.nodeid]['father']]:
                    gain_lastlast = self.tg[self.nodeid]['p'] - \
                                    self.tg[self.tg[self.tg[self.nodeid]['father']]['father']]['p']

            # budget
            budget = 1 - self.countnode * 1.0 / self.evalcount
            # depth
            depth = self.tg[self.nodeid]['d']
            depth = abs(depth)

            self.state = np.concatenate([qsa_rep, np.copy(self.transform), np.copy(act_node_visit), np.copy(self.action_count),
                                         np.array(gain_each), np.array(action_count_root),
                                         np.array([gain_last, gain_lastlast, budget, depth])], axis=None)

            allperf = np.array([self.tg[i]['p'] for i in range(self.countnode)])
            startnode = allperf.argmax()
            self.best_pfm = allperf.max()

            # print('best-----------',self.best_pfm)
            if self.globalreward:
                reward = allperf.max() - self._init_pfm

            self.best_seq = []
            while "father" in self.tg[startnode] and startnode > 0:
                lastnode = self.tg[startnode]['father']
                for key in self.tg[lastnode]:
                    if key != 'fid' and self.tg[lastnode][key] == startnode:
                        self.best_seq.insert(0, key)
                        break
                startnode = lastnode
            # print(self.best_seq, np.array([self.tg[i]['p'] for i in range(self.countnode) ]).max())

            return self.state, reward

    def _QSA(self):
        """
        根据给定的参数和数据集，它通过对特征进行分箱和统计计算，生成表示当前状态的向量。
        :return:
        """
        global feat_0, feat_1
        if self.opt_type == 'o1':
            if self.tasktype == "C":
                feat_0 = self.feature[self.y == 0]
                feat_1 = self.feature[self.y == 1]
            elif self.tasktype == "R":
                median = np.median(self.y)
                feat_0 = self.feature[self.y < median]
                feat_1 = self.feature[self.y >= median]

            minval, maxval = feat_0.min(), feat_0.max()
            if abs(maxval - minval) < 1e-8:
                QSA0 = [0] * self.binsize
            else:
                bins = np.arange(minval, maxval, (maxval - minval) * 1.0 / self.binsize)[1:self.binsize]
                QSA0 = np.bincount(np.digitize(feat_0, bins)).astype(float) / len(feat_0)

            minval, maxval = feat_1.min(), feat_1.max()
            if abs(maxval - minval) < 1e-8:
                QSA1 = [0] * self.binsize
            else:
                bins = np.arange(minval, maxval, (maxval - minval) * 1.0 / self.binsize)[1:self.binsize]
                QSA1 = np.bincount(np.digitize(feat_1, bins)).astype(float) / len(feat_1)
            QSA = np.concatenate([QSA0, QSA1])

        elif self.opt_type == 'o2':
            QSA = []
            for i in range(2):
                if self.tasktype == "C":
                    feat_0 = self.feature[:, i][self.y == 0]
                    feat_1 = self.feature[:, i][self.y == 1]
                elif self.tasktype == 'R':
                    median = np.median(self.y)
                    feat_0 = self.feature[:, i][self.y < median]
                    feat_1 = self.feature[:, i][self.y >= median]

                minval, maxval = feat_0.min(), feat_0.max()
                if abs(maxval - minval) < 1e-8:
                    QSA0 = [0] * self.binsize
                else:
                    bins = np.arange(minval, maxval, (maxval - minval) * 1.0 / self.binsize)[1:self.binsize]
                    QSA0 = np.bincount(np.digitize(feat_0, bins)).astype(float) / len(feat_0)

                minval, maxval = feat_1.min(), feat_1.max()
                if abs(maxval - minval) < 1e-8:
                    QSA1 = [0] * self.binsize
                else:
                    bins = np.arange(minval, maxval, (maxval - minval) * 1.0 / self.binsize)[1:self.binsize]
                    QSA1 = np.bincount(np.digitize(feat_1, bins)).astype(float) / len(feat_1)
                QSA.append(QSA0)
                QSA.append(QSA1)
            QSA = np.concatenate(QSA)
        return QSA

    def reset(self):
        self.dataset = self.origin_dataset
        self.feature = self.origin_feat
        self._init()

    def fe(self, operators, feat_id):
        # target = self.dataset[:,-1]
        # self.dataset = pd.DataFrame(np.copy(self.dataset[:, :-1]))
        # print(operators)
        # print('fe',feat_id)
        if type(feat_id) is int:
            new_feat_id = self._pretrf_mapper[feat_id]

            if new_feat_id != -1:
                feature = self.origin_dataset[:, new_feat_id]
            else:
                feature = None
        else:
            new_feat_id_a = self._pretrf_mapper[feat_id[0]]
            new_feat_id_b = self._pretrf_mapper[feat_id[1]]
            new_feat_id = [new_feat_id_a, new_feat_id_b]
            if new_feat_id_a != -1 and new_feat_id_b != -1:
                feature = self.origin_dataset[:, new_feat_id]
            else:
                feature = None

        for operator in operators:
            if type(feat_id) is int:
                if operator in {'square', 'tanh', 'round'}:
                    feature = getattr(np, operator)(feature)
                elif operator == 'log':
                    vmin = feature.min()
                    feature = np.log(feature - vmin + 1) if vmin < 1 else np.log(feature)
                elif operator == "sqrt":
                    vmin = feature.min()
                    feature = np.sqrt(feature - vmin) if vmin < 0 else np.sqrt(feature)
                elif operator == "mmn":
                    mmn = MinMaxScaler()
                    feature = mmn.fit_transform(feature[:, np.newaxis]).flatten()
                elif operator == "sigmoid":
                    feature = (1 + getattr(np, 'tanh')(feature / 2)) / 2
                elif operator == 'zscore':
                    if np.var(feature) != 0:
                        feature = stats.zscore(feature)
                    else:
                        feature = None
                else:
                    feature = None
            else:
                if operator == "sum":
                    feature = feature.sum(axis=1)
                elif operator == 'diff':
                    feature = feature[:, 0] * feature[:, 1]
                elif operator == "product":
                    feature = feature[:, 0] * feature[:, 1]
                elif operator == 'divide':
                    over = feature[:, 1]
                    while np.any(over == 0):
                        over = over + 1e-5
                    feature = feature[:, 0] / over
                else:
                    feature = None

        if len(operators) > 0 and feature is not None and operators[0] != 'fs':
            feature = np.nan_to_num(feature)
            feature = np.clip(feature, -math.sqrt(3.4e38), math.sqrt(3.4e38))
            self.origin_dataset = np.insert(self.origin_dataset, -1, feature, axis=1)
        if len(operators) > 0 and operators[0] == 'fs':
            self.origin_dataset = np.delete(self.origin_dataset, new_feat_id, axis=1)
            if type(feat_id) is int:
                self._pretrf_mapper[feat_id] = -1
                for i in range(feat_id, len(self._pretrf_mapper)):
                    if self._pretrf_mapper[i] >= 1:
                        self._pretrf_mapper[i] -= 1
            else:
                for feat in feat_id:
                    self._pretrf_mapper[feat] = -1
                for i in range(min(feat_id), len(self._pretrf_mapper)):
                    if self._pretrf_mapper[i] >= 1:
                        self._pretrf_mapper[i] -= 1
                for i in range(max(feat_id), len(self._pretrf_mapper)):
                    if self._pretrf_mapper[i] >= 1:
                        self._pretrf_mapper[i] -= 1

# Simple feed forward neural network
class Model(nn.Module):
    def __init__(self, opt_size, input_size, name, meta=False, update_lr=1e-3, meta_lr=0.001, num_updates=1, maml=True,qsasize=200):
        super(Model, self).__init__()
        self.input_size = input_size
        self.opt_size = self.dim_output = opt_size
        self.dim_hidden = [128, 128, 64]
        self.skip = 1
        self.qsasize = qsasize
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.num_updates = num_updates
        self.size = opt_size
        self.inputs = torch.empty((0, self.input_size), dtype=torch.float32)
        self.Q_next = torch.tensor([], dtype=torch.float32)
        self.action = torch.tensor([], dtype=torch.int32)
        self.inputsa = torch.empty((0, 0, self.input_size), dtype=torch.float32)
        self.inputsb = torch.empty((0, 0, self.input_size), dtype=torch.float32)
        self.Q_nexta = torch.empty((0,0),dtype=torch.float32)
        self.Q_nextb = torch.empty((0, 0), dtype=torch.float32)
        self.actiona = torch.empty((0, 0), dtype=torch.int32)
        self.actionb = torch.empty((0, 0), dtype=torch.int32)
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.num_updates = num_updates
        self.size = opt_size
        self.input_size = input_size
        self.loss_func = self.mse
        self.weights = self.construct_fc_weights()
        self.network()
        if maml:
            self.construct_model()

    def mse(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true, reduction='sum')

    def construct_fc_weights(self):
        factor = 1
        weights = nn.ParameterDict()

        weights['w1'] = nn.Parameter(torch.Tensor(self.input_size, self.dim_hidden[0]).normal_(
            std=math.sqrt(factor / ((self.input_size + self.dim_hidden[0]) / 2))))
        weights['b1'] = nn.Parameter(torch.zeros(self.dim_hidden[0]))

        for i in range(1, len(self.dim_hidden)):
            weights['w' + str(i + 1)] = nn.Parameter(torch.Tensor(self.dim_hidden[i - 1], self.dim_hidden[i]).normal_(
                std=math.sqrt(factor / ((self.dim_hidden[i - 1] + self.dim_hidden[i]) / 2))))
            weights['b' + str(i + 1)] = nn.Parameter(torch.zeros(self.dim_hidden[i]))

        if self.skip == 1:
            weights['skip' + str(len(self.dim_hidden) + 1 - 1)] = nn.Parameter(
                torch.Tensor(self.input_size - self.qsasize, self.dim_hidden[-1]).normal_(
                    std=math.sqrt(factor / ((self.input_size - self.qsasize + self.dim_hidden[-1])))))
        elif self.skip == 0:
            weights['skip' + str(len(self.dim_hidden) + 1)] = nn.Parameter(
                torch.Tensor(self.input_size - self.qsasize, self.dim_output).normal_(
                    std=math.sqrt(factor / ((self.input_size - self.qsasize + self.dim_output)))))

        else:
            pass

        weights['w' + str(len(self.dim_hidden) + 1)] = nn.Parameter(
            torch.Tensor(self.dim_hidden[-1], self.dim_output).normal_(
                std=math.sqrt(factor / ((self.dim_hidden[-1] + self.dim_output) / 2))))

        return weights

    def forward(self, inp, weights):
        hidden = nn.functional.relu(torch.matmul(inp, weights['w1']) + weights['b1'])

        if self.skip == 1:
            for i in range(1, len(self.dim_hidden) - 1):
                hidden = nn.functional.relu(torch.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)])

            hiddenp1 = torch.matmul(hidden, weights['w' + str(i + 1 + 1)]) + weights['b' + str(i + 1 + 1)]
            hiddenp2 = torch.matmul(inp[:, self.qsasize:], weights['skip' + str(len(self.dim_hidden) + 1 - 1)])
            hidden = nn.functional.relu(hiddenp1 + hiddenp2)
            Q_ = torch.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)])

        elif self.skip == 0:
            for i in range(1, len(self.dim_hidden)):
                hidden = nn.functional.relu(torch.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)])

            Q_ = torch.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + \




