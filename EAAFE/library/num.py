from sklearn.preprocessing import MinMaxScaler
import numpy as np

class tf_num():

    def __init__(self):
        pass

    def transfrom(self, fe1):
        pass


class num_sqrt(tf_num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1):
        return np.sqrt(np.abs(fe1))

    def __repr__(self):
        return 'num_sqrt'

class num_minmaxscaler(tf_num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1):
        scaler = MinMaxScaler()
        return np.squeeze(scaler.fit_transform(np.reshape(fe1, [-1, 1])))

    def __repr__(self):
        return 'num_minmaxscaler'

class num_log(tf_num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1):
        while (np.any(fe1 == 0)):
            fe1 = fe1 + 1e-3
        return np.log(np.abs(fe1))

    def __repr__(self):
        return 'num_log'

class num_reciprocal(tf_num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1):
        while (np.any(fe1 == 0)):
            fe1 = fe1 + 1e-3
        return np.reciprocal(fe1)

    def __repr__(self):
        return 'num_reciprocal'