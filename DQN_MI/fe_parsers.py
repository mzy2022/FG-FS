from fe_operations import OPS

def parse_actions(actions_ops,  c_ops,d_ops,df_c_encode,df_d_encode,n_c_features,n_d_features):
    operations_c = len(OPS["arithmetic"]) * n_c_features + len(OPS["value_convert"])
    operations_d = len(OPS["discrete"]) * n_d_features + 1
    add = []
    subtract = []
    multiply = []
    divide = []
    combine = []
    nunique = []
    value_c_convert = {}
    len_c = n_c_features

    for index,ops in enumerate(actions_ops):
        if 0 <= index < len_c:
            if operations_c <= 7:
                break
            ops = ops % operations_c
            if 0 <= ops < n_c_features:
                add.append([index, ops])
            elif n_c_features <= ops < (2 * n_c_features):
                subtract.append([index, ops - n_c_features])
            elif (2 * n_c_features) <= ops < (3 * n_c_features):
                multiply.append([index, ops - n_c_features * 2])
            elif (3 * n_c_features) <= ops < (4 * n_c_features):
                divide.append([index, ops - n_c_features * 3])
            else:
                value_c_convert[index] = [ops - n_c_features * 4]

        else:
            if operations_d <= 0:
                break
            x = ops % operations_d
            if 0 <= x < n_d_features:
                combine.append([index - len_c, x])
            elif n_d_features <= x < 2 * n_d_features:
                nunique.append([index - len_c, x - n_d_features])
            else:
                combine.append([index - len_c, 'None'])

    action_all = [{"add": add}, {"subtract": subtract}, {"multiply": multiply}, {"divide": divide},
                  {"combine": combine},
                  {"value_c_convert": value_c_convert}, {"nunique": nunique}]


    return action_all
