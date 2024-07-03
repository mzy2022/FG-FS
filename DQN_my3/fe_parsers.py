from fe_operations import OPS


def parse_actions(actions_ops,actions_features, actions_otp, c_ops, d_ops, df_c_encode, df_d_encode, n_c_features, n_d_features,c_features,d_features):
    double_nums = len(OPS["arithmetic"])
    single_nums = len(OPS["value_convert"])
    dis_nums = len(OPS["discrete"])
    operations_c = double_nums + single_nums
    operations_d = dis_nums
    add = []
    subtract = []
    multiply = []
    divide = []
    combine = []
    nunique = []
    value_c_convert = {}
    len_c = n_c_features

    for index, (ops,feature, opt) in enumerate(zip(actions_ops,actions_features, actions_otp)):
        real_otp = opt % 3
        if 0 <= index < len_c:
            if n_c_features <= 0:
                break
            feature = feature % n_c_features
            ops = ops % operations_c
            if ops == 0:
                add.append([index, feature, real_otp])
            elif ops == 1:
                subtract.append([index, feature, real_otp])
            elif ops == 2:
                multiply.append([index, feature, real_otp])
            elif ops == 3:
                divide.append([index, feature, real_otp])
            else:
                value_c_convert[index] = [ops - 4, real_otp]
        else:
            if n_d_features <= 0:
                break
            feature = feature % n_d_features
            x = ops % operations_d
            if x == 0:
                combine.append([index - len_c, feature, real_otp])
            elif x == 1:
                nunique.append([index - len_c, feature, real_otp])

    action_all = [{"add": add}, {"subtract": subtract}, {"multiply": multiply}, {"divide": divide},
                  {"combine": combine},
                  {"value_c_convert": value_c_convert}, {"nunique": nunique}]

    return action_all
