from .feature_generate_memory import *
from .utils_memory import categories_to_int

class Update_data():
    def __init__(self,state_c,state_d,pipline_data):
        self.c_col_eval = []
        self.d_col_eval = []
        self.c_col_model = []
        self.d_col_model = []
        self.continuous = state_c.copy()
        self.combine = state_d.copy()
        self.ori_cols_continuous = pipline_data['continuous_data'].copy()
        self.ori_cols_combine = pipline_data['discrete_data'].copy()
        self.Candidate_features = pipline_data['dataframe']



    def process_data(self, actions):
        self.specials = {'concat': [], 'replace': [], 'delete': []}
        for action in actions:
            self.__refresh_continuous_actions__(action)
            self.arithmetic_operations()
            self.single_fe_operations()
            self.feature_combine()

        return self.continuous, self.combine,self.specials

    def __refresh_continuous_actions__(self, actions):
        self.value_convert = actions['value_c_convert'] if 'value_c_convert' in actions else {}  # dict
        self.add_ = actions['add'] if 'add' in actions else []
        self.subtract_ = actions['subtract'] if 'subtract' in actions else []
        self.multiply_ = actions['multiply'] if 'multiply' in actions else []
        self.divide_ = actions['divide'] if 'divide' in actions else []
        self.combine_ = actions["combine"] if "combine" in actions else []
        self.nunique_ = actions["nunique"] if "nunique" in actions else []


    def get_candidate_feature(self,name, is_test=False):
        if is_test:
            return None
        if name in self.Candidate_features.columns.tolist():
            return self.Candidate_features[name]
        else:
            return None

    def arithmetic_operations(self):
        delete_index = []
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
                x = (abs(col1 - col2)).sum()

                if abs(x) < 0.1 and operation in ['subtract','divide']:
                    # new_fe = np.array(self.continuous.iloc[:, col1_index])
                    # name = self.continuous.iloc[:, col1_index].name
                    continue
                else:
                    name = col1.name + '_' + operation + '_' + col2.name
                    # res1 = self.get_candidate_feature(name)
                    # if res1 is not None:
                    #     new_fe = res1.values
                    # if operation in ['add', 'multiply']:
                        # name2 = col2.name + '_' + operation + '_' + col1.name
                        # res2 = self.get_candidate_feature(name2)
                        # if res2 is not None:
                        #     new_fe = res2.values
                        # else:
                        # new_fe = globals()[operation](col1.values, col2.values)
                        # if special_op != 1:
                        #     self.Candidate_features[name] = new_fe
                    # else:
                    new_fe = globals()[operation](col1.values, col2.values)
                        # if special_op != 1:
                        #     self.Candidate_features[name] = new_fe

                # self.continuous = self.continuous.copy()
                if special_op == 0:
                    self.continuous[name] = new_fe.reshape(-1)
                    self.specials['concat'].append([col1.values, new_fe.flatten()])
                elif special_op == 2:
                    delete_index.append(col1_index)
                    self.specials['delete'].append([col1.values, new_fe.flatten()])
                else:
                    self.continuous[col1.name] = new_fe.reshape(-1)
                    self.continuous = self.continuous.rename(columns={col1.name: name})
                    self.specials['replace'].append([col1.values, new_fe.flatten()])
        # if 0 < len(delete_index) < self.continuous.shape[1]:
        #     self.continuous_drop_list.extend(delete_index)


    def single_fe_operations(self):
        delete_index = []
        operations = ["abss", 'square', 'inverse', 'log', 'sqrt', 'power3', 'None']
        for index, operation in self.value_convert.items():
            if self.continuous.shape[1] == 0:
                continue
            ori_col = self.continuous.iloc[:, index].copy()
            if operation[0] == 6:
                new_fe = np.array(self.continuous.iloc[:, index])
                name = self.continuous.iloc[:, index].name
            else:
                name = ori_col.name + '_' + operations[operation[0]]
                # res = self.get_candidate_feature(name)
                # if res is not None:
                #     new_fe = res.values
                # else:
                new_fe = globals()[operations[operation[0]]](ori_col.values)
                # if operations[1] != 1:
                #     self.Candidate_features[name] = new_fe
            if operation[1] == 0:
                self.continuous[name] = new_fe.reshape(-1).copy()
                self.specials['concat'].append([ori_col.values, new_fe.flatten()])
            elif operation[1] == 1:
                self.continuous[ori_col.name] = new_fe.reshape(-1)
                self.continuous = self.continuous.rename(columns={ori_col.name: name})
                self.specials['replace'].append([ori_col.values, new_fe.flatten()])
            else:
                delete_index.append(index)
                self.specials['delete'].append([ori_col.values, new_fe.flatten()])


    def feature_combine(self):
        delete_index = []
        operations = ["combine", "nunique"]
        feature_informations = [self.combine_, self.nunique_]
        for i, feature_information in enumerate(feature_informations):
            if len(feature_information) == 0:
                continue
            if self.combine.shape[1] == 0:
                continue

            elif operations[i] == "combine":
                for actions in feature_information:
                    if actions[1] == 'None':
                        continue
                    index1 = index = actions[0]
                    ori_fe1 = self.combine.iloc[:, index1]
                    ori_fe2 = self.ori_cols_combine.iloc[:, actions[1]]
                    name = ori_fe1.name + '_combine_' + ori_fe2.name
                    feasible_values = {}
                    cnt = 0
                    for x in np.unique(ori_fe1):
                        for y in np.unique(ori_fe2):
                            feasible_values[str(int(x)) + str(int(y))] = cnt
                            cnt += 1
                    new_fe = generate_combine_fe(ori_fe1.values, ori_fe2, feasible_values)

                    if actions[2] == 0:
                        self.combine[name] = new_fe.reshape(-1)
                    elif actions[2] == 1:
                        self.combine[ori_fe1.name] = new_fe.reshape(-1)
                        self.combine = self.combine.rename(columns={ori_fe1.name: name})
                    else:
                        delete_index.append(index)

            elif operations[i] == "nunique":
                for actions in feature_information:
                    index = actions[0]
                    ori_fe1 = self.combine.iloc[:, index]
                    ori_fe2 = self.ori_cols_combine.iloc[:, actions[1]]
                    name = ori_fe1.name + '_nunique_' + ori_fe2.name
                    new_fe = get_nunique_feature(ori_fe1 ,ori_fe2)

                    # self.Candidate_features[name] = new_fe
                    if actions[2] == 0:
                        self.combine[name] = new_fe.reshape(-1)
                        self.specials['concat'].append([ori_fe1.values, new_fe.flatten()])
                    elif actions[2] == 1:
                        self.combine[ori_fe1.name] = new_fe.reshape(-1)
                        self.combine = self.combine.rename(columns={ori_fe1.name: name})
                        self.specials['replace'].append([ori_fe1.values, new_fe.flatten()])
                    else:
                        delete_index.append(index)
                        self.specials['delete'].append([ori_fe1.values, new_fe.flatten()])
        # if 0 < len(delete_index) < self.combine.shape[1]:
        #     self.combine_drop_list.extend(delete_index)
