import xgboost
import lightgbm
import catboost
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


def rf_classify():
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    return model


def rf_regression():
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    return model


def lr_classify(penalty='l2'):
    model = LogisticRegression(class_weight='balanced', tol=0.0005, C=0.1,
                               max_iter=10000, penalty=penalty)
    return model


def lr_regression():
    model = LinearRegression()
    return model


def xgb_classify():
    # model = xgboost.XGBClassifier(max_depth=6, learning_rate="0.1",
    #                               n_estimators=600, verbosity=0, subsample=0.8,
    #                               colsample_bytree=0.8, use_label_encoder=False, scale_pos_weight=1, n_jobs=n_jobs)
    model = xgboost.XGBClassifier(n_estimators=10, random_state=0)
    return model


def xgb_regression():
    # model = xgboost.XGBRegressor(max_depth=6, learning_rate="0.1",
    #                              n_estimators=600, verbosity=0, subsample=0.8,
    #                              colsample_bytree=0.8, use_label_encoder=False, scale_pos_weight=1, n_jobs=n_jobs)
    model = xgboost.XGBRegressor(n_estimators=10, random_state=0)
    return model


def lgb_classify():
    model = lightgbm.LGBMClassifier(n_estimators=10, random_state=0)
    return model


def lgb_regression():
    model = lightgbm.LGBMRegressor(n_estimators=10, random_state=0)
    return model


def cat_classify():
    model = catboost.CatBoostClassifier(n_estimators=10, random_state=0)
    return model


def cat_regression():
    model = catboost.CatBoostRegressor(n_estimators=10, random_state=0)
    return model


def rf_pre_classify():
    model = RandomForestClassifier(n_estimators=600, max_depth=8, class_weight='balanced', random_state=42)
    return model


def rf_pre_regression():
    model = RandomForestRegressor(n_estimators=600, max_depth=8, random_state=42)
    return model


model_fuctions = {"lr_regression": lr_regression, "lr_classify": lr_classify, "rf_regression": rf_regression,
                  "rf_classify": rf_classify, "xgb_regression": xgb_regression, "xgb_classify": xgb_classify,
                  "lgb_regression": lgb_regression, "lgb_classify": lgb_classify,"cat_regression": cat_regression,
                  "cat_classify": cat_classify, "rf_pre_regression": rf_pre_regression, "rf_pre_classify": rf_pre_classify,}
