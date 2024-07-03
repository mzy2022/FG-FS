import numpy as np
import pandas as pd

from DQN_my3.process_data.feature_generate_memory import binning_with_tree

OPS = {
    "arithmetic": ["add", "subtract", "multiply", "divide"],
    "value_convert": ["abss", 'square', 'inverse', 'log', 'sqrt', 'power3'],
    "special_ops": ["reserve", "replace", "delete"],
    "discrete": ["combine","nunique"]
}


def get_ops(n_features_c, n_features_d,c_columns,d_columns):
    c_ops = []
    d_ops = []
    sps = []
    c_features = []
    d_features = []
    arithmetic = OPS["arithmetic"]
    value_convert_c = OPS["value_convert"]
    specials = OPS["special_ops"]
    discrete = OPS["discrete"]
    if n_features_c == 0:
        c_ops = []
    elif n_features_c == 1:
        c_ops.extend(value_convert_c)
    else:
        c_ops.extend(arithmetic)
        c_ops.extend(value_convert_c)
    if n_features_d != 0:
        d_ops.extend(discrete)
    else:
        d_ops = []

    max_len = max(len(c_ops), len(d_ops))
    if len(d_ops) < max_len and len(d_ops) != 0:
        d_ori_ops = d_ops
        while len(d_ops) < max_len:
            d_ops.extend(d_ori_ops)
        d_ops = d_ops[:max_len]
    if len(c_ops) < max_len and len(c_ops) != 0:
        c_ori_ops = c_ops
        while len(c_ops) < max_len:
            c_ops.extend(c_ori_ops)
        c_ops = c_ops[:max_len]

    max_features_len = max(len(c_columns),len(d_columns))
    if len(c_columns) <= max_features_len and len(c_columns) != 0:
        c_ori_columns = c_columns
        while len(c_features) < max_features_len:
            c_features.extend(c_ori_columns)
        c_features = c_features[:max_features_len]
    if len(d_columns) <= max_features_len and len(d_columns) != 0:
        d_ori_columns = d_columns
        while len(d_features) < max_features_len:
            d_features.extend(d_ori_columns)
        d_features = d_features[:max_features_len]

    while len(sps) < max_len:
        sps.extend(specials)
    sps = sps[:max_len]

    return c_ops, d_ops, sps,c_features,d_features


def get_binning_df(args, df, c_columns, d_columns, mode):
    new_df = pd.DataFrame()
    new_c_columns = []
    new_d_columns = []
    label = df.loc[:, args.target]
    if mode == 'classify':
        for col in c_columns:
            ori_fe = np.array(df[col])
            label = np.array(label)
            new_fe = binning_with_tree(ori_fe, label)
            new_name = 'bin_' + col
            new_df[new_name] = new_fe
            new_d_columns.append(new_name)
        for col in d_columns:
            new_df[col] = df[col]
            new_d_columns.append(col)
        for col in c_columns:
            new_df[col] = df[col]
            new_c_columns.append(col)
    else:
        for col in c_columns:
            new_df[col] = df[col]
            new_c_columns.append(col)
        for col in d_columns:
            new_df[col] = df[col]
            new_d_columns.append(col)
    new_df[args.target] = label
    return new_df, new_c_columns, new_d_columns


# def get_binning_df(args, df, c_columns, d_columns, mode):
#     new_df = pd.DataFrame()
#     new_c_columns = []
#     new_d_columns = []
#     label = df.loc[:, args.target]
#     if mode == 'classify':
#         for col in c_columns:
#             ori_fe = np.array(df[col])
#             label = np.array(label)
#             # new_fe = binning_with_tree(ori_fe, label)
#             # new_name = 'bin_' + col
#             # new_df[new_name] = new_fe
#             # new_d_columns.append(new_name)
#         for col in d_columns:
#             new_df[col] = df[col]
#             new_c_columns.append(col)
#         for col in c_columns:
#             new_df[col] = df[col]
#             new_c_columns.append(col)
#     else:
#         for col in c_columns:
#             new_df[col] = df[col]
#             new_c_columns.append(col)
#         for col in d_columns:
#             new_df[col] = df[col]
#             new_d_columns.append(col)
#     new_df[args.target] = label
#     return new_df, new_c_columns, new_d_columns
