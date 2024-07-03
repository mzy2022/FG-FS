import logging
import os
import random
import time

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR, SVC
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from DQN_my2.process_data.update_data import Update_data
from fe_operations import get_binning_df
from fe_operations import get_ops
from feature_engineer import DQN_ops, DQN_otp
from feature_engineer.replay import Replay_ops, Replay_otp
from feature_engineer.training_ops import sample, multiprocess_reward, sample_update
from feature_engineer.worker import Worker
from metric_evaluate import metric_fuctions
from metric_evaluate import rae_score
from model_evaluate import *
from process_data import Feature_type_recognition
from utils import log_dir
from DQN_my2.feature_engineer.training_ops import get_reward


class AutoFE:
    def __init__(self, input_data: pd.DataFrame, args):
        times = time.strftime('%Y%m%d-%H%M')
        log_path = fr"./logs/{args.file_name}_{times}"
        log_dir(log_path)
        self.log_path = log_path
        logging.info(args)
        logging.info(f'File name: {args.file_name}')
        logging.info(f'Data shape: {input_data.shape}')
        # Fixed random seed
        self.seed = args.seed
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        # os.environ["PYTHONHASHSEED"] = str(self.seed)
        self.info_ = {}
        self.best_score = 0
        self.info_['target'] = args.target
        self.info_['file_name'] = args.file_name
        self.info_['mode'] = args.mode
        self.info_['metric'] = args.metric
        self.info_['model'] = args.model
        self.train_size = args.train_size
        self.shuffle = args.shuffle
        self.split = args.split_train_test
        self.info_['c_columns'] = args.c_columns
        self.info_['d_columns'] = args.d_columns

        for col in input_data.columns:
            col_type = input_data[col].dtype
            if col_type != 'object':
                input_data[col].fillna(0, inplace=True)
            else:
                input_data[col].fillna('unknown', inplace=True)
        self.ori_df = input_data

        self.is_cuda, self.device = None, None
        self.set_cuda(args.cuda)

    def set_cuda(self, cuda):
        if cuda == 'False':
            self.device = 'cpu'
        else:
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        logging.info(f"Use device:{self.device}")
        return

    def fit_data(self, args):
        if self.split:
            train_data, test_data = split_train_test(self.ori_df, self.info_['target'], self.info_['mode'],
                                                     self.train_size,
                                                     self.seed,
                                                     self.shuffle)
            train_data.reset_index(inplace=True, drop=True)
            test_data.reset_index(inplace=True, drop=True)
            df = train_data
        else:
            df = self.ori_df
        c_columns, d_columns = self.info_['c_columns'], self.info_['d_columns']
        target, mode, model, metric = self.info_['target'], self.info_['mode'], self.info_['model'], self.info_[
            'metric']
        label = df.loc[:, target]
        new_df, new_c_columns, new_d_columns = get_binning_df(args, df, c_columns, d_columns, mode)
        if self.split:
            test_df, new_test_c_columns, new_test_d_columns = get_binning_df(args, test_data, c_columns, d_columns,
                                                                             mode)
            df_test_c_encode, df_test_d_encode = test_df.loc[:, new_test_c_columns], test_df.loc[:, new_test_d_columns]
            pipline_test_data = {'dataframe': test_df,
                                 'continuous_columns': new_test_c_columns,
                                 'discrete_columns': new_test_d_columns,
                                 'continuous_data': df_test_c_encode,
                                 'discrete_data': df_test_d_encode,
                                 'label_name': target,
                                 'mode': mode
                                 }
            scores = get_test_score(new_df.iloc[:, :-1], test_df.iloc[:, :-1], new_df.loc[:, target],
                                    test_df.loc[:, target], args, mode, model, metric)
            logging.info(f'test_train_score={np.mean(scores)}')
        # df_c_encode = pd.DataFrame()
        # for col in new_c_columns:
        #     df_c_encode[col] = normalization(new_df[col].values).reshape(-1)
        # new_df = pd.concat([df_c_encode,new_df[new_d_columns],new_df[target]],axis=1)
        n_features_c, n_features_d = len(new_c_columns), len(new_d_columns)
        c_ops, d_ops, sps = get_ops(n_features_c, n_features_d)

        score_b, scores_b,scores_knn,scores_log,scores_xgb,scores_light,scores_cat = self._get_cv_baseline(new_df, args, mode, model, metric)
        score_ori, scores_ori,scores_knn1,scores_log1,scores_xgb1,scores_light1,scores_cat1 = self._get_cv_baseline(df, args, mode, model, metric)

        logging.info(f'score_b={score_b}')
        logging.info(f'score_ori={score_ori}')
        logging.info(f"knn{scores_knn1}")
        # logging.info(f"svm{scores_svm1}")
        logging.info(f"log{scores_log1}")
        logging.info(f"xgb{scores_xgb1}")
        logging.info(f"light{scores_light1}")
        logging.info(f"cat{scores_cat1}")

        # processing data
        df_c_encode, df_d_encode = new_df.loc[:, new_c_columns + [target]], new_df.loc[:, new_d_columns + [target]]
        # x_d_onehot, df_d_labelencode = new_df.loc[:, new_d_columns], new_df.loc[:, new_d_columns]
        ori_continuous_data = df.loc[:, c_columns]
        df_t, df_t_norm = new_df.loc[:, target], new_df.loc[:, target]
        feature_nums = n_features_c + n_features_d
        data_nums = new_df.shape[0]
        operations_c = len(c_ops)
        operations_d = len(d_ops)
        d_model = args.d_model
        d_k = args.d_k
        d_v = args.d_v
        d_ff = args.d_ff
        n_heads = args.n_heads
        batch_size = args.batch_size
        memory_size = args.memory
        hidden_size = args.hidden_size
        self.memory_ops = Replay_ops(batch_size, memory_size)
        self.memory_otp = Replay_otp(batch_size, memory_size)
        self.dqn_ops = DQN_ops(args, data_nums, feature_nums, operations_c, operations_d, d_model, d_k, d_v, d_ff,
                               n_heads, self.memory_ops, self.device)
        self.dqn_otp = DQN_otp(args, operations_c, operations_d, hidden_size, d_model, self.memory_otp, self.device)
        self.steps_done = 0
        pipline_data = {'dataframe': new_df,
                        'continuous_columns': new_c_columns,
                        'discrete_columns': new_d_columns,
                        'continuous_data': df_c_encode,
                        'discrete_data': df_d_encode,
                        'label_name': target,
                        'mode': mode,
                        'ori_continuous_data': ori_continuous_data
                        }

        self.workers_top5 = []
        score_list = []
        ## 强化学习轮次
        for epoch in tqdm(range(args.epochs)):
            print(torch.cuda.max_memory_allocated() / 1024 ** 2)
            epoch_best_score = 0
            workers = []
            score_epoch_list = []
            logging.info(f"now_epoch{epoch}")
            for _ in range(args.episodes):
                worker = Worker(args)
                init_state = torch.from_numpy(new_df.values).float().transpose(0, 1)
                worker.states = [init_state]
                worker.actions, worker.steps = [], []
                worker.features, worker.ff = [], []
                worker.scores_b = scores_b
                worker.best_score = 0
                workers.append(worker)

            ###worker 数量
            for i in range(args.steps_num):
                # 得到经过两个agent之后的actions，states
                workers_dqn = sample(args, self.dqn_ops, self.dqn_otp, pipline_data, df_c_encode, df_d_encode,
                                     df_t_norm, c_ops, d_ops, i, workers, self.device, self.steps_done)
                # 得到reward
                workers_reward = multiprocess_reward(args, pipline_data, workers_dqn, scores_b, mode, model, metric,
                                                     df_t.values)
                # 得到经过两个agent之后的actions_，states_
                workers_dqn_ = sample_update(args, pipline_data, self.dqn_ops, self.dqn_otp, df_c_encode, df_d_encode,
                                             df_t_norm, c_ops, d_ops, workers_reward, self.steps_done, self.device)

                workers = workers_dqn_

                self.dqn_ops.store_transition(args, workers_dqn_)
                self.dqn_otp.store_transition(args, workers_dqn_)


                for i, worker in enumerate(workers_dqn_):
                    worker_x = workers_dqn_[i]
                    # logging.info(f"worker{i + 1} epoch{epoch},shape:{worker.c_d.shape}")
                    worker = Worker(args)
                    worker.accs = worker_x.accs
                    worker.scores = worker_x.scores
                    worker.ff = worker_x.ff
                    worker.x_c_d = worker_x.x_c_d
                    # epoch_best_score += worker.scores.mean()
                    score_epoch_list.append(worker.scores.mean())


                    if np.mean(worker.scores) > self.best_score:
                        if self.split:
                            y = test_df.loc[:, target]
                            pipline_ff = Update_data(df_test_c_encode, df_test_d_encode, pipline_test_data)
                            for fe in worker.ff:
                                df_new_test_c_encode, df_new_test_d_encode = pipline_ff.process_data(fe)
                            if df_new_test_c_encode.shape[0] == 0:
                                x = df_new_test_d_encode
                            elif df_new_test_d_encode.shape[0] == 0:
                                x = df_new_test_c_encode
                            else:
                                x = pd.concat([df_new_test_c_encode, df_new_test_d_encode], axis=1)
                            df_train = worker.x_c_d
                            df_test = x
                            label_train = new_df.loc[:, target]
                            label_test = y
                            scores = get_test_score(df_train, df_test, label_train, label_test, args, mode, model,
                                                    metric)
                            # acc, cv, scores = get_reward(x, y, args, scores_b, mode, model, metric)
                            print(f"%%%%%%%%%%%%%%{np.mean(scores)}************")

                        self.best_score = np.mean(worker.scores)
                        logging.info(f"epoch:{epoch}_new_best_score{self.best_score}")

                        xxx = pd.concat([worker.x_c_d, label], axis=1)
                        df = pd.DataFrame(xxx)
                        df.to_csv(f"{self.log_path}/test.csv", index=False)

                        ########鲁棒性
                        # if mode == 'classify':
                        #     knn = KNeighborsClassifier(n_neighbors=3)
                        #     svm_classifier = SVC(kernel='linear')
                        #     log_cls = LogisticRegression()
                        #     model_xgb = xgboost.XGBClassifier(n_estimators=10, random_state=0)
                        #     model_light = lightgbm.LGBMClassifier(n_estimators=10, random_state=0)
                        #     model_cat = catboost.CatBoostClassifier(n_estimators=10, random_state=0)
                        #     scores_knn = cross_val_score(knn, worker_x.x_c_d, label, scoring='f1_micro', cv=5).mean()
                        #     # scores_svm = cross_val_score(svm_classifier, worker_x.x_c_d, label, scoring='f1_micro', cv=5).mean()
                        #     scores_log = cross_val_score(log_cls, worker_x.x_c_d, label, scoring='f1_micro', cv=5).mean()
                        #     scores_xgb = cross_val_score(model_xgb, worker_x.x_c_d, label, scoring='f1_micro', cv=5).mean()
                        #     scores_light = cross_val_score(model_light, worker_x.x_c_d, label, scoring='f1_micro', cv=5).mean()
                        #     scores_cat = cross_val_score(model_cat, worker_x.x_c_d, label, scoring='f1_micro', cv=5).mean()
                        #
                        #     logging.info(f"epoch:{epoch}_knn{scores_knn}")
                        #     # logging.info(f"epoch:{epoch}_svm{scores_svm}")
                        #     logging.info(f"epoch:{epoch}_log{scores_log}")
                        #     logging.info(f"epoch:{epoch}_xgb{scores_xgb}")
                        #     logging.info(f"epoch:{epoch}_light{scores_light}")
                        #     logging.info(f"epoch:{epoch}_cat{scores_cat}")
                        # else:
                            # knn_reg = KNeighborsRegressor(n_neighbors=3)
                            # svm_regressor = SVR(kernel='linear')
                            # log_reg = LinearRegression()
                            # model_xgb = xgboost.XGBRegressor(n_estimators=10, random_state=0)
                            # model_light = lightgbm.LGBMRegressor(n_estimators=10, random_state=0)
                            # model_cat = catboost.CatBoostRegressor(n_estimators=10, random_state=0)
                            #
                            # rae_score1 = make_scorer(sub_rae, greater_is_better=True)

                            # scores_knn = cross_val_score(knn_reg, worker_x.x_c_d, label, cv=5, scoring=rae_score1).mean()
                            # scores_svm = cross_val_score(svm_regressor, worker_x.x_c_d, label, cv=5, scoring=rae_score1)
                            # scores_log = cross_val_score(log_reg, worker_x.x_c_d, label, cv=5, scoring=rae_score1).mean()
                            # scores_xgb = cross_val_score(model_xgb, worker_x.x_c_d, label, cv=5, scoring=rae_score1).mean()
                            # scores_light = cross_val_score(model_light, worker_x.x_c_d, label, cv=5, scoring=rae_score1).mean()
                            # scores_cat = cross_val_score(model_cat, worker_x.x_c_d, label, cv=5, scoring=rae_score1).mean()
                            #
                            # logging.info(f"epoch:{epoch}_knn{scores_knn}")
                            # logging.info(f"epoch:{epoch}_svm{scores_svm}")
                            # logging.info(f"epoch:{epoch}_log{scores_log}")
                            # logging.info(f"epoch:{epoch}_xgb{scores_xgb}")
                            # logging.info(f"epoch:{epoch}_light{scores_light}")
                            # logging.info(f"epoch:{epoch}_cat{scores_cat}")







                #     self.workers_top5.append(worker)
                #
                # self.workers_top5.sort(key=lambda worker1: worker1.scores.mean(), reverse=True)
                # self.workers_top5 = self.workers_top5[0:5]
                #
                # for i in range(1):
                #     logging.info(f"top_{i + 1}:score:{self.workers_top5[i].scores.mean()}")

                self.dqn_ops.learn(args, workers_dqn_, self.device)
                self.dqn_otp.learn(args, workers_dqn_, self.device)
            newlist = sorted(score_epoch_list,reverse=True)
            new_scores = np.mean(newlist[:5])
            score_list.append(new_scores)
            print(score_list)
        logging.info(score_list)

    def _get_cv_baseline(self, df: pd.DataFrame, args, mode, model, metric):
        target = self.info_["target"]

        model = model_fuctions[f"{model}_{mode}"]()
        args.seed = None
        if mode == "classify":
            my_cv = StratifiedKFold(n_splits=args.cv, random_state=args.seed)
        else:
            my_cv = KFold(n_splits=args.cv, random_state=args.seed)

        X = df.drop(columns=[target])
        y = df[target]
        scores = []
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        if mode == "classify":
            if metric == 'f1':
                clf = RandomForestClassifier(n_estimators=10, random_state=0)
                # clf = SVC(kernel='linear')
                scores = cross_val_score(clf, X, y, scoring='f1_micro', cv=5)

                knn = KNeighborsClassifier(n_neighbors=3)
                svm_classifier = SVC(kernel='linear')
                log_cls = LogisticRegression()
                model_xgb = xgboost.XGBClassifier(n_estimators=10, random_state=0)
                model_light = lightgbm.LGBMClassifier(n_estimators=10, random_state=0)
                model_cat = catboost.CatBoostClassifier(n_estimators=10, random_state=0)
                scores_knn = cross_val_score(knn, X, y, scoring='f1_micro', cv=5).mean()
                # scores_svm = cross_val_score(svm_classifier,X, y, scoring='f1_micro', cv=5).mean()
                scores_log = cross_val_score(log_cls, X, y, scoring='f1_micro', cv=5).mean()
                scores_xgb = cross_val_score(model_xgb,X, y, scoring='f1_micro', cv=5).mean()
                scores_light = cross_val_score(model_light, X, y, scoring='f1_micro', cv=5).mean()
                scores_cat = cross_val_score(model_cat, X, y, scoring='f1_micro', cv=5).mean()

                # scores = f1(X, y)
            elif metric == 'auc':
                auc_scorer = make_scorer(roc_auc_score, needs_proba=True, average="macro", multi_class="ovo")
                scores = cross_val_score(model, X, y, scoring=auc_scorer, cv=my_cv, error_score="raise")
        else:
            if metric == 'mae':
                scores = cross_val_score(model, X, y, cv=my_cv, scoring='neg_mean_absolute_error')
            elif metric == 'mse':
                scores = cross_val_score(model, X, y, cv=my_cv, scoring='neg_mean_squared_error')
            elif metric == 'r2':
                scores = cross_val_score(model, X, y, cv=my_cv, scoring='r2')
            elif metric == 'rae':
                model = RandomForestRegressor(n_estimators=10, random_state=0)
                rae_score1 = make_scorer(sub_rae, greater_is_better=True)
                scores = cross_val_score(model, X, y, cv=my_cv, scoring=rae_score1)

                knn_reg = KNeighborsRegressor(n_neighbors=3)
                svm_regressor = SVR(kernel='linear')
                log_reg = LinearRegression()
                model_xgb = xgboost.XGBRegressor(n_estimators=10, random_state=0)
                model_light = lightgbm.LGBMRegressor(n_estimators=10, random_state=0)
                model_cat = catboost.CatBoostRegressor(n_estimators=10, random_state=0)



                scores_knn = cross_val_score(knn_reg, X, y, cv=5, scoring=rae_score1).mean()
                # scores_svm = cross_val_score(svm_regressor, X, y, cv=5, scoring=rae_score1)
                scores_log = cross_val_score(log_reg, X, y, cv=5, scoring=rae_score1).mean()
                scores_xgb = cross_val_score(model_xgb,X, y, cv=5, scoring=rae_score1).mean()
                scores_light = cross_val_score(model_light, X, y, cv=5, scoring=rae_score1).mean()
                scores_cat = cross_val_score(model_cat,X, y, cv=5, scoring=rae_score1).mean()

                # scores = rae(X,y)
        return np.array(scores).mean(), scores,scores_knn,scores_log,scores_xgb,scores_light,scores_cat


def f1(X, y):
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    clf = RandomForestClassifier(random_state=0)
    f1_list = []
    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train, test in skf.split(X, y):
        X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        f1_list.append(f1_score(y_test, y_predict, average='weighted'))
    return np.array(f1_list)


def rae(X, y):
    reg = RandomForestRegressor(random_state=0)
    rae_list = []
    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    for train, test in kf.split(X):
        X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
        reg.fit(X_train, y_train)
        y_predict = reg.predict(X_test)
        rae_list.append(1 - relative_absolute_error(y_test, y_predict))
    return np.array(rae_list)


def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(y_test) - y_test))
    return error


def normalization(col: list or np.ndarray) -> np.array:
    col = np.array(col)
    mu = np.mean(col, axis=0)
    sigma = np.std(col, axis=0)
    if sigma == 0:
        return col.reshape(-1, 1)
    else:
        scaled = ((col - mu) / sigma)
    return scaled.reshape(-1, 1)


def split_train_test(df, target, mode, train_size, seed, shuffle):
    """
    Split data into training set and test set

    :param df: pd.DataFrame, origin data
    :param d_columns: a list of the names of discrete columns
    :param target: str, label name
    :param mode: str, classify or regression
    :param seed: int, to fix random seed
    :param train_size: float
    :return: df_train_val, df_test
    """
    # for col in d_columns:
    #     new_fe = merge_categories(df[col].values)
    #     df[col] = new_fe

    if mode == "classify":
        df_train_val, df_test = train_test_split(df, train_size=train_size, random_state=seed,
                                                 stratify=df[target], shuffle=shuffle)
    else:
        df_train_val, df_test = train_test_split(df, train_size=train_size, random_state=seed, shuffle=shuffle)

    # df_train_val = df_train_val.copy()
    # for col in d_columns:
    #     new_fe = merge_categories(df_train_val[col].values)
    #     df_train_val[col] = new_fe

    return df_train_val, df_test


def get_test_score(df_train, df_test, label_train, label_test, args, mode, model, metric):
    model = model_fuctions[f"{model}_{mode}"]()
    model.fit(df_train, label_train)
    # pred = model.predict(df_test)
    score = metric_fuctions[metric](model, df_test, label_test, label_train)
    return score


# def sub_rae(y, y_hat):
#     y = np.array(y).reshape(-1)
#     y_hat = np.array(y_hat).reshape(-1)
#     y_mean = np.mean(y)
#     rae = np.sum([np.abs(y_hat[i] - y[i]) for i in range(len(y))]) / np.sum(
#         [np.abs(y_mean - y[i]) for i in range(len(y))])
#     res = 1 - rae
#     return res

def sub_rae(y, y_hat):
    y = np.array(y).reshape(-1)
    y_hat = np.array(y_hat).reshape(-1)
    y_mean = np.mean(y)

    # 计算每个预测值与实际值之间的绝对误差
    absolute_errors = np.abs(y_hat - y)

    # 计算每个实际值与均值之间的绝对误差
    mean_errors = np.abs(y_mean - y)

    # 计算RAE，然后计算其补值
    rae = np.sum(absolute_errors) / np.sum(mean_errors)
    res = 1 - rae

    return res
