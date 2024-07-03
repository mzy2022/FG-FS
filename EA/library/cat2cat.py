import numpy as np
import pandas as pd

class tf_cat2cat():
    def __init__(self):
        pass

    def transform(self, fe1, fe2):
        pass

class cat2cat_get_count_feature(tf_cat2cat):
    def __init__(self):
        super().__init__()

    def get_hash_feature(self,feature1,feature2):
        return feature1 + feature2 + feature1 * feature2

    def get_count_feature(self, feature):
        return (pd.DataFrame(feature).fillna(0)).groupby([0])[0].transform('count').values

    def transform(self, fe1, fe2):
        return self.get_count_feature(self.get_hash_feature(fe1,fe2))

    def __repr__(self):
        return 'cat2cat_get_count_feature'

class cat2cat_get_nunique_feature(tf_cat2cat):
    def __init__(self):
        super().__init__()

    def get_nunique_feature(self,feature1,feature2):
        feature = np.concatenate([np.reshape(feature1, [-1, 1]), np.reshape(feature2, [-1, 1])],
                                 axis=1)
        return (pd.DataFrame(feature).fillna(0)).groupby(0)[1].transform('nunique').values

    def transform(self, fe1, fe2):
        return self.get_nunique_feature(fe1, fe2)

    def __repr__(self):
        return 'cat2cat_get_nunique_feature'
