from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import LinearSVC, LinearSVR


class ModelBase(object):
    def __init__(self):
        self.n_estimators = 10
        self.random_state = 0

    @staticmethod
    def xgb_classify(random_state=None):
        model = XGBClassifier(max_depth=6, learning_rate="0.1",
                              n_estimators=200, verbosity=0, subsample=0.8, random_state=random_state,
                              colsample_bytree=0.8, use_label_encoder=False)
        return model

    @staticmethod
    def xgb_regression(random_state=None):
        model = XGBRegressor(max_depth=4, learning_rate="0.05", min_child_weight=1, n_estimators=300, verbosity=1,
                             subsample=0.8,
                             random_state=random_state, colsample_bytree=0.8, use_label_encoder=False,
                             scale_pos_weight=1, n_jobs=1)
        return model

    @staticmethod
    def rf_classify(random_state=None):
        model = RandomForestClassifier(random_state=0, n_estimators=10, n_jobs=-1)
        return model

    @staticmethod
    def rf_regeression(random_state=None):
        model = RandomForestRegressor(random_state=0, n_estimators=10, n_jobs=-1)
        return model


    @staticmethod
    def lr_classify(random_state=None):
        model = LogisticRegression(penalty='l2', dual=False, tol=0.0001,
                                   C=1, fit_intercept=True, intercept_scaling=1.0,
                                   random_state=random_state, class_weight=None)

        return model

    @staticmethod
    def lr_regression(random_state=None):
        model = Lasso(tol=0.0001, max_iter=1000, random_state=random_state, alpha=0.1)
        return model

    @staticmethod
    def svm_liner_svc(random_state=None):
        model = LinearSVC(random_state=random_state)
        return model

    @staticmethod
    def svm_liner_svr(random_state=None):
        model = LinearSVR(random_state=random_state)
        return model
