from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
class algorithm:
    def __init__(self):
        pass
    def model(self):
        pass
    def hyper_params(self):
        pass
    def get_model_name(self):
        pass
    def fit(self, X, y):
        pass
    def predit(self, X):
        pass

class random_forest_regressor(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = RandomForestRegressor(**params)
        self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.predict(X=X)


class random_forest_classifier(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = RandomForestClassifier()
        self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.predict(X=X)

class linearRegression(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = LinearRegression()
        self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.predict(X=X)


class logisticRegression(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = LogisticRegression()
        self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.predict(X=X)


class LGBMRegressor(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = lgb.LGBMRegressor()
        self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.predict(X=X)


class LGBMClassifier(algorithm):

    def __init__(self, params=None):
        super().__init__()
        self.model = lgb.LGBMClassifier()
        self.model.set_params(**params)

    def model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predit(self, X):
        return self.model.predict(X=X)



