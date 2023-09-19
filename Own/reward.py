import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, f1_score
from model_based import ModelBase
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
import warnings
from Own.Evolutionary_FE.DNA_Fitness import relative_absolute_error

warnings.filterwarnings("ignore")

def sub_rae(y, y_hat):
    y = np.array(y).reshape(-1)
    y_hat = np.array(y_hat).reshape(-1)
    y_mean = np.mean(y)
    rae = np.sum([np.abs(y_hat[i] - y[i]) for i in range(len(y))]) / np.sum(
        [np.abs(y_mean - y[i]) for i in range(len(y))])
    res = 1 - rae
    return res

class GetReward(object):
    def __init__(self, args):
        self.args = args
        self.target_col = args.target_col
        self.task_type = args.task_type
        self.eval_method = args.eval_method


    def get_model(self):
        if self.args.model == 'rf':
            return self.get_rf_model(None)
        elif self.args.model == 'lr':
            return self.get_lr_model()
        elif self.args.model == 'xgb':
            return self.get_xgb_model()



    def get_lr_model(self):
        if self.task_type == 'classifier':
            model = ModelBase.lr_classify()
        elif self.task_type == 'regression':
            model = ModelBase.lr_regression()
        else:
            logging.info(f'er')
            model = None
        return model

    def get_svm_model(self):
        # choose model
        if self.task_type == 'classifier':
            model = ModelBase.svm_liner_svc()
        elif self.task_type == 'regression':
            model = ModelBase.svm_liner_svr()
        else:
            logging.info(f'er')
            model = None
        return model

    def get_rf_model(self, hyper_param):
        #
        if self.task_type == 'classifier':
            model = ModelBase.rf_classify(self.args.seed)
        elif self.task_type == 'regression':
            model = ModelBase.rf_regeression(self.args.seed)
        else:
            logging.info(f'er')
            model = None
        if hyper_param is not None and model is not None:
            model.set_params(**hyper_param)
        return model

    def get_xgb_model(self):
        #
        if self.task_type == 'classifier':
            model = ModelBase.xgb_classify()
        elif self.task_type == 'regression':
            model = ModelBase.xgb_regression()
        else:
            logging.info(f'er')
            model = None
        return model

    def k_fold_score_(self, search_fes, search_label):
        global score_list, scoring_name
        if self.task_type == 'classifier':
            if self.eval_method == 'auc':
                scoring_name = "accuracy"
            elif self.eval_method == 'f1_score':
                scoring_name = "f1_micro"
            score_list = cross_val_score(self.get_lr_model(), search_fes, search_label, scoring=scoring_name,cv=5)

        else:
            if self.eval_method == 'sub_rae':
                subrae = make_scorer(sub_rae, greater_is_better=True)
                score_list = cross_val_score(self.get_model(), search_fes, search_label, scoring=subrae, cv=5)

            elif self.eval_method == 'rmse':
                score_list = -cross_val_score(self.get_model(), search_fes, search_label,
                                              scoring="neg_mean_squared_error", cv=5)
                for i in range(len(score_list)):
                    score_list[i] = np.sqrt(score_list[i])
            elif self.eval_method == 'mae':
                score_list = -cross_val_score(self.get_model(), search_fes, search_label,
                                              scoring="neg_mean_absolute_error",cv=5)

            score = np.mean(score_list)
            return score

    def downstream_task_new(self,search_fes, search_label,task_type):
        X = search_fes
        y = search_label
        if task_type == 'cls':
            clf = RandomForestClassifier(random_state=0)
            f1_list = []
            skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            for train, test in skf.split(X, y):
                X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
                clf.fit(X_train, y_train)
                y_predict = clf.predict(X_test)
                f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            return np.mean(f1_list)
        elif task_type == 'reg':
            reg = RandomForestRegressor(random_state=0)
            rae_list = []
            kf = KFold(n_splits=5, random_state=0, shuffle=True)
            for train, test in kf.split(X):
                X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
                reg.fit(X_train, y_train)
                y_predict = reg.predict(X_test)
                rae_list.append(1 - relative_absolute_error(y_test, y_predict))
            return np.mean(rae_list)
        else:
            return -1