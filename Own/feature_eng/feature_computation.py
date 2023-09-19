import numpy as np
from scipy.special import expit

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

O1 = ['sqrt', 'square', 'sin', 'cos', 'tanh', 'abs'
      'power3','round','sigmoid', 'log','reciprocal','none']
O2 = ['+', '-', '*', '/']


def cube(x):
    return x ** 3


def none(x):
    return x
def justify_operation_type(op):
    if op == 'sqrt':
        op = np.sqrt
    elif op == 'none':
        op = none
    elif op == 'abs':
        op = np.abs
    elif op == 'power3':
        op = cube
    elif op == 'round':
        op = np.round
    elif op == 'reciprocal':
        op = np.reciprocal
    elif op == 'square':
        op = np.square
    elif op == 'sin':
        op = np.sin
    elif op == 'cos':
        op = np.cos
    elif op == 'tanh':
        op = np.tanh
    elif op == '+':
        op = np.add
    elif op == '-':
        op = np.subtract
    elif op == '/':
        op = np.divide
    elif op == '*':
        op = np.multiply
    elif op == 'exp':
        op = np.exp
    elif op == 'sigmoid':
        op = expit
    elif op == 'log':
        op = np.log
    else:
        print('Please check your operation!')
    return op

def single_combine_tuple(combine_tuple,ori_cols):
    combine_array = combine_noonehot(ori_cols, combine_tuple)
    return combine_array

def parser_single_opslist_combine(self, ops_list, frame):
    col_idx0 = self.all_discrete_col.index(ops_list[0])
    ops = ops_list[-2]
    ori_ops = ops_list[-1]
    if ops == 'None':
        return frame.iloc[:, [col_idx0]].values

    if len(ops_list) == 4:
        if ops == 'combine':
            col_idx1 = self.all_discrete_col.index(ops_list[1])
            combineops_param = ((ops_list[0], ops_list[1]), frame.iloc[:, [col_idx0, col_idx1]].values)
            arr = self.single_combine_tuple(*combineops_param)

            if ori_ops in ['concat', 'concat_END']:
                colname = len(list(frame))
            elif ori_ops in ['replace', 'replace_END']:
                colname = col_idx0
            else:
                raise ValueError(f'ori_ops : {ori_ops} not define')
            return arr

        else:
            raise ValueError(f'ops : {ops} not define')

    else:
        raise ValueError(f'combine opslist length: {ops_list} must be 4')


def combine_noonehot(self, ori_fes, fe_names):
    fe_names = list(fe_names)
    cb_df = pd.DataFrame(ori_fes, columns=fe_names, dtype='int').astype(str)
    uniuqe_idx = cb_df.groupby(fe_names).count().reset_index()[fe_names]
    uniuqe_idx = uniuqe_idx.sort_values(by=fe_names, ascending=True).astype(str)
    uniuqe_idx['keys'] = uniuqe_idx[fe_names].apply(lambda x: ''.join(x), axis=1)
    uniuqe_idx['coding'] = range(1, len(uniuqe_idx) + 1)
    col_unique_dict = dict(zip(uniuqe_idx['keys'], uniuqe_idx['coding']))
    cb_df = pd.merge(cb_df, uniuqe_idx[fe_names + ['coding']], on=fe_names, how='left')
    combine_col = cb_df['coding'].values.astype(int)
    self.feature_eng_combine_dict[tuple(fe_names)] = col_unique_dict
    return combine_col.reshape(-1, 1)
