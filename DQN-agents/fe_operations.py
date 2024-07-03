import numpy as np
import pandas as pd

from PPO_transformer_lstm.process_data.feature_generate_memory import binning_with_tree

OPS = {
    "arithmetic": ["add", "subtract", "multiply", "divide"],
    "value_convert": ["abss", 'square', 'inverse', 'log', 'sqrt', 'power3','None'],
    "special_ops": ["reserve","replace"]
}


def get_ops(n_features_c, n_features_d):
    c_ops = []
    d_ops = []
    sps = []
    arithmetic = OPS["arithmetic"]
    value_convert_c = OPS["value_convert"]
    specials = OPS["special_ops"]
    if n_features_c == 0:
        c_ops = []
    elif n_features_c == 1:
        c_ops.extend(value_convert_c)
    else:
        for i in range(4):
            op = arithmetic[i]
            for j in range(n_features_c):
                c_ops.append(op)
        c_ops.extend(value_convert_c)
    if n_features_d != 0:
        d_ops = ["combine" for _ in range(n_features_d)]
        d_ops.append("None")
    else:
        d_ops = []

    max_len = max(len(c_ops),len(d_ops))
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

    while len(sps) < max_len:
        sps.extend(specials)

    sps = sps[:max_len]

    return c_ops,d_ops,sps



def get_binning_df(df,c_columns,d_columns,mode):
    new_df = pd.DataFrame()
    new_c_columns = []
    new_d_columns =[]
    label = df.iloc[:,-1]
    if mode == 'classify':
        for col in c_columns:
            ori_fe = np.array(df[col])
            label = np.array(label)
            new_fe = binning_with_tree(ori_fe,label)
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
    new_df['label'] = label
    return new_df,new_c_columns,new_d_columns

