import numpy as np
import pandas as pd

class tf_num2num():
    def __init__(self):
        pass

    def transform(self, fe1, fe2):
        pass


class num2num_add(tf_num2num):
    def __init__(self):
        super().__init__()

        def transform(self, fe1, fe2):
            return np.squeeze(fe1 + fe2)

        def __repr__(self):
            return 'num2num_add'

class num2num_sub(tf_num2num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2):
        return np.squeeze(fe1 - fe2)

    def __repr__(self):
        return 'num2num_sub'

class num2num_mul(tf_num2num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2):
        return np.squeeze(fe1 * fe2)

    def __repr__(self):
        return 'num2num_mul'


class num2num_div(tf_num2num):
    def __init__(self):
        super().__init__()

    def transform(self, fe1, fe2):
        while (np.any(fe2 == 0)):
            fe2 = fe2 + 1e-3
        return np.squeeze(fe1 / fe2)

    def __repr__(self):
        return 'num2num_div'