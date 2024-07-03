import numpy as np
import pandas as pd

class tf_cat2num():

    def __init__(self):
        pass

    def transform(self, fe1, fe2):
        pass


class cat2num_get_mean_feature(tf_cat2num):
    def __init__(self):
        super().__init__()
    def get_cat2num_mean_feature(self, feature1, feature2):
        feature = np.concatenate([np.reshape(feature1, [-1, 1]), np.reshape(feature2, [-1, 1])],
                                 axis=1)
        return (pd.DataFrame(feature).fillna(0)).groupby(0)[1].transform('mean').values

    def transform(self, fe1, fe2):
        return self.get_cat2num_mean_feature(fe1, fe2)

    def __repr__(self):
        return 'cat2num_get_mean_feature'