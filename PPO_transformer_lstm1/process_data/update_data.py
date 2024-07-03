import pandas as pd

from .feature_generate_memory import *
from .utils_memory import categories_to_int
from PPO_transformer_lstm1.fe_operations import OPS
class Update_data():
    def __init__(self,state_c,state_d,pipline_data):
        self.c_col_eval = []
        self.d_col_eval = []
        self.c_col_model = []
        self.d_col_model = []
        self.continuous = state_c
        self.combine = state_d
        self.new_continuous = pd.DataFrame()
        self.new_combine = pd.DataFrame()
        self.ori_cols_continuous = pipline_data['continuous_data']
        self.ori_cols_combine = pipline_data['discrete_data']
        self.Candidate_features = pipline_data['dataframe']



    def process_data(self, actions):
        for action in actions:
            self.__refresh_continuous_actions__(action)
            self.arithmetic_operations()
            self.single_fe_operations()
            self.feature_combine()

        return self.new_continuous, self.new_combine

    def __refresh_continuous_actions__(self, actions):
        self.value_convert = actions['value_c_convert'] if 'value_c_convert' in actions else {}  # dict
        self.add_ = actions['add'] if 'add' in actions else []
        self.subtract_ = actions['subtract'] if 'subtract' in actions else []
        self.multiply_ = actions['multiply'] if 'multiply' in actions else []
        self.divide_ = actions['divide'] if 'divide' in actions else []
        self.combine_ = actions["combine"] if "combine" in actions else []


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
                col1_index, col2_index = col_index_tuple[:2]
                if self.continuous.shape[1] == 0:
                    continue
                # col1_index表示当前特征的索引,col2_index表示原始特征的索引
                col1 = self.continuous.iloc[:, col1_index]
                col2 = self.ori_cols_continuous.iloc[:, col2_index]
                if col1.equals(col2):
                    new_fe = np.array(self.continuous.iloc[:, col1_index])
                    name = self.continuous.iloc[:, col1_index].name
                else:
                    name = col1.name + '_' + operation + '_' + col2.name
                    # res1 = self.get_candidate_feature(name)
                    # if res1 is not None:
                    #     new_fe = res1.values
                    if operation in ['add', 'multiply']:
                        name2 = col2.name + '_' + operation + '_' + col1.name
                        # res2 = self.get_candidate_feature(name2)
                        # if res2 is not None:
                        #     new_fe = res2.values
                        # else:
                        new_fe = globals()[operation](col1.values, col2.values)
                            # self.Candidate_features[name] = new_fe
                    else:
                        new_fe = globals()[operation](col1.values, col2.values)
                        # self.Candidate_features[name] = new_fe
                    # self.continuous = self.continuous.copy()

                self.new_continuous[name] = new_fe.reshape(-1)


    def single_fe_operations(self):
        delete_index = []
        operations = ["abss", 'square', 'inverse', 'log', 'sqrt', 'power3','None']
        for index, operation in self.value_convert.items():
            if self.continuous.shape[1] == 0:
                continue
            ori_col = self.continuous.iloc[:, index].copy()
            if operation[0] == 6:
                new_fe = np.array(self.continuous.iloc[:, index])
                name = self.continuous.iloc[:,index].name
            else:
                name = ori_col.name + '_' + operations[operation[0]]
                # res = self.get_candidate_feature(name)
                # if res is not None:
                #     new_fe = res.values
                # else:
                new_fe = globals()[operations[operation[0]]](ori_col.values)
                    # self.Candidate_features[name] = new_fe

            self.new_continuous[name] = new_fe.reshape(-1)



    def feature_combine(self):
        delete_index = []
        for actions in self.combine_:
            index = actions[0]
            if self.combine.shape[1] == 0:
                continue
            if actions[1] == 'None':
                new_fe = np.array(self.combine.iloc[:, index])
                name = self.combine.iloc[:, index].name
            else:
                ori_fe1 = self.combine.iloc[:, index]
                ori_fe2 = self.ori_cols_combine.iloc[:,actions[1]]
                name = ori_fe1.name + '_combine_' + ori_fe2.name
            # res1 = self.get_candidate_feature(name)
            # if res1 is not None:
            #     new_fe = res1.values
            # else:
            #     name2 = ori_fe2.name + '_combine_' + ori_fe1.name
            #     res2 = self.get_candidate_feature(name2)
            #     if res2 is not None:
            #         new_fe = res2.values
            #     else:
                feasible_values = {}
                cnt = 0
                for x in np.unique(ori_fe1):
                    for y in np.unique(ori_fe2):
                        feasible_values[str(int(x)) + str(int(y))] = cnt
                        cnt += 1
                new_fe = generate_combine_fe(ori_fe1.values, ori_fe2,feasible_values)
                    # self.Candidate_features[name] = new_fe

            self.new_combine[name] = new_fe.reshape(-1)

