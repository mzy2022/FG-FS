import logging

from .feature_generate_memory import *
from .utils_memory import categories_to_int


class Pipeline(object):
    """
    The pipeline of feature processing, including normalization, applying actions, etc.
    """

    def __init__(self, args, bins_thresh=8):
        self.ori_dataframe = args['dataframe']
        self.Candidate_features = args['dataframe']
        self.continuous_columns = args['continuous_columns']
        self.discrete_columns = args['discrete_columns']
        self.mode = args['mode']
        self.bins_thresh = bins_thresh
        self.c_col_eval = []
        self.d_col_eval = []
        self.c_col_model = []
        self.d_col_model = []
        self.continuous_drop_list = []
        self.combine_drop_list = []
        if self.mode == 'classify':
            self.label = categories_to_int(self.ori_dataframe[args['label_name']].values.reshape(-1),-1)
            self.ori_dataframe = self.ori_dataframe.copy()
            self.ori_dataframe[args['label_name']] = self.label
        else:
            self.label = self.ori_dataframe[args['label_name']].values
        self.refresh_states()

    def get_candidate_feature(self,name, is_test=False):
        if is_test:
            return None
        if name in self.Candidate_features.columns.tolist():
            return self.Candidate_features[name]
        else:
            return None

    def refresh_states(self):
        logging.debug(f'Start discrete_fe_2_int_type')
        self.continuous = self.ori_dataframe[self.continuous_columns]
        self.ori_cols_continuous = self.ori_dataframe[self.continuous_columns]


        self.combine = self.ori_dataframe[self.discrete_columns]
        self.ori_cols_combine = self.ori_dataframe[self.discrete_columns]

    def __refresh_continuous_actions__(self, actions):
        self.value_convert = actions['value_convert'] if 'value_convert' in actions else {}  # dict
        self.add_ = actions['add'] if 'add' in actions else []
        self.subtract_ = actions['subtract'] if 'subtract' in actions else []
        self.multiply_ = actions['multiply'] if 'multiply' in actions else []
        self.divide_ = actions['divide'] if 'divide' in actions else []
        self.combine_ = actions["combine"] if "combine" in actions else []


    def process_data(self, actions):
        self.combine_drop_list = []
        self.continuous_drop_list = []
        for action in actions:
            self.__refresh_continuous_actions__(action)
            self.arithmetic_operations()
            self.single_fe_operations()
            self.feature_combine()
        # self.delete()

        return self.continuous, self.combine

    def delete(self):
        if self.continuous.shape[1] > len(self.continuous_drop_list):
            self.continuous = self.continuous.drop(self.continuous.columns[self.continuous_drop_list], axis=1)
        if self.combine.shape[1] > len(self.combine_drop_list):
            self.combine = self.combine.drop(self.combine.columns[self.combine_drop_list], axis=1)

    def arithmetic_operations(self):
        delete_index = []
        past_num = self.continuous.shape[1]
        operations = ['add', 'subtract', 'multiply', 'divide']
        feature_informations = [self.add_, self.subtract_, self.multiply_, self.divide_]
        for i, feature_information in enumerate(feature_informations):
            if len(feature_information) == 0:
                continue
            combine_feature_tuples_list = feature_information
            operation = operations[i]
            for col_index_tuple in combine_feature_tuples_list:
                special_op = col_index_tuple[2]
                col1_index, col2_index = col_index_tuple[:2]
                if self.continuous.shape[1] == 0:
                    continue
                # col1_index表示当前特征的索引,col2_index表示原始特征的索引
                col1 = self.continuous.iloc[:, col1_index]
                col2 = self.ori_cols_continuous.iloc[:, col2_index]
                if col1.equals(col2):
                    continue
                name = col1.name + '_' + operation + '_' + col2.name
                res1 = self.get_candidate_feature(name)
                if res1 is not None:
                    new_fe = res1.values
                elif operation in ['add', 'multiply']:
                    name2 = col2.name + '_' + operation + '_' + col1.name
                    res2 = self.get_candidate_feature(name2)
                    if res2 is not None:
                        new_fe = res2.values
                    else:
                        new_fe = globals()[operation](col1.values, col2.values)
                        if special_op != 1:
                            self.Candidate_features[name] = new_fe
                else:
                    new_fe = globals()[operation](col1.values, col2.values)
                    if special_op != 1:
                        self.Candidate_features[name] = new_fe

                self.continuous = self.continuous.copy()
                if special_op == 0:
                    self.continuous[name] = new_fe.reshape(-1)
                elif special_op == 2:
                    delete_index.append(col1_index)
                else:
                    self.continuous[col1.name] = new_fe.reshape(-1)
                    self.continuous = self.continuous.rename(columns={col1.name:name})
        if 0 < len(delete_index) < self.continuous.shape[1]:
            self.continuous_drop_list.extend(delete_index)




    def single_fe_operations(self):
        delete_index = []
        for index, operations in self.value_convert.items():
            if self.continuous.shape[1] == 0:
                continue
            ori_col = self.continuous.iloc[:, index].copy()
            if operations[0] == "None":
                continue
            else:
                name = ori_col.name + '_' + operations[0]
                res = self.get_candidate_feature(name)
                if res is not None:
                    new_fe = res.values
                else:
                    new_fe = globals()[operations[0]](ori_col.values)
                    if operations[1] != 1:
                        self.Candidate_features[name] = new_fe
                if operations[1] == 0:
                    self.continuous[name] = new_fe.reshape(-1).copy()
                elif operations[1] == 1:
                    self.continuous[ori_col.name] = new_fe.reshape(-1)
                    self.continuous = self.continuous.rename(columns={ori_col.name: name})
                else:
                    delete_index.append(index)
        if 0 < len(delete_index) < self.continuous.shape[1]:
            self.continuous_drop_list.extend(delete_index)

    def feature_combine(self):
        delete_index = []
        for actions in self.combine_:
            index = actions[0]
            if self.combine.shape[1] == 0:
                continue
            if actions[1] == 'None':
                continue
            ori_fe1 = self.combine.iloc[:, index]
            ori_fe2 = self.ori_cols_combine.iloc[:,actions[1]]
            name = ori_fe1.name + '_combine_' + ori_fe2.name
            res1 = self.get_candidate_feature(name)
            if res1 is not None:
                new_fe = res1.values
            else:
                name2 = ori_fe2.name + '_combine_' + ori_fe1.name
                res2 = self.get_candidate_feature(name2)
                if res2 is not None:
                    new_fe = res2.values
                else:
                    feasible_values = {}
                    cnt = 0
                    for x in np.unique(ori_fe1):
                        for y in np.unique(ori_fe2):
                            feasible_values[str(int(x)) + str(int(y))] = cnt
                            cnt += 1
                    new_fe = generate_combine_fe(ori_fe1.values, ori_fe2,feasible_values)

                    self.Candidate_features[name] = new_fe
            if actions[2] == 0:
                self.combine[name] = new_fe.reshape(-1)
            elif actions[2] == 1:
                self.combine[ori_fe1.name] = new_fe.reshape(-1)
                self.combine = self.combine.rename(columns={ori_fe1.name: name})
            else:
                delete_index.append(index)
        if 0 < len(delete_index) < self.combine.shape[1]:
            self.combine_drop_list.extend(delete_index)
