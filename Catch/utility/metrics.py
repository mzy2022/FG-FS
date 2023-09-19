from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("..")
from feature_engineering.model_base import ModelBase

def get_model(model_type,task_type,hyper_param):
    if model_type == 'lr':
        if task_type == 'classifier':
            model = ModelBase.lr_classify()
        elif task_type == 'regression':
            model = ModelBase.lr_regression()
        else:
            model = None
    elif model_type == 'svm':
        if model_type == 'classifier':
            model = ModelBase.svm_liner_svc()
        elif task_type == 'regression':
            model = ModelBase.svm_liner_svr()
        else:
            model = None
    elif model_type == 'lr':
        if task_type == 'classifier':
            model = ModelBase.lr_classify()
        elif task_type == 'regression':
            model = ModelBase.lr_regression()
        else:
            model = None
    elif model_type == 'rf':
        if task_type == 'classifier':
            model = ModelBase.rf_classify(0)
        elif task_type == 'regression':
            model = ModelBase.rf_regeression(0)
        else:
            model = None
        if hyper_param is not None and model is not None:
            model.set_params(**hyper_param)
    elif model_type == 'xgb':
        if task_type == 'classifier':
            model = ModelBase.xgb_classify()
        elif task_type == 'regression':
            model = ModelBase.xgb_regression()
        else:
            model = None
    else:
        model = None
    return model



def calculate_f1_score(df,target_col,model_type,task_type,hyper_param=None,pos_label = 0,average='weighted',test_size=0.2,random_state=42):
    df.columns = df.columns.astype(str)
    X = df.drop(target_col,axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = get_model(model_type,task_type,hyper_param)
    model.fit(X_train, y_train,hyper_param)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred,pos_label=pos_label,average=average)
    return f1

