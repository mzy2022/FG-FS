from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import feature_generate_memory, utils_memory


def split_train_test(df, d_columns, target, mode, train_size, seed, shuffle):
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
        df_train_val, df_test = train_test_split(df, train_size=train_size, random_state=seed, stratify=df[target],
                                                 shuffle=shuffle)
    else:
        df_train_val, df_test = train_test_split(df, train_size=train_size, random_state=seed, shuffle=shuffle)

    # df_train_val = df_train_val.copy()
    # for col in d_columns:
    #     new_fe = merge_categories(df_train_val[col].values)
    #     df_train_val[col] = new_fe

    return df_train_val, df_test
