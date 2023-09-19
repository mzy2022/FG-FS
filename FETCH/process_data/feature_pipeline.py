import logging

import numpy as np

from .feature_generate_memory import *
from .utils_memory import categories_to_int
from .utils_memory import combine_feature_tuples
from .memory_no_encode import Memory
import pandas as pd

# global Candidate_features
Candidate_features = pd.DataFrame()


# 根据给定的特征名称获取候选特征
def get_candidate_feature(name, is_test=False):
    if is_test:
        return None
    if name in Candidate_features.columns.tolist():
        return Candidate_features[name]
    else:
        return None


class Pipeline(object):

    def __init__(self, args, bins_thresh=8):
        self.ori_dataframe = args['dataframe']  # 这个属性保存原始的dataframe，所有step不再修改
        self.continuous_columns = args['continuous_columns']
        self.discrete_columns = args['discrete_columns']
        self.mode = args['mode']
        self.isvalid = args['isvalid']
        self.memory = args['memory']
        self.bins_thresh = bins_thresh
        self.terminate_list = []
        self.c_col_eval = []
        self.d_col_eval = []
        self.c_col_model = []
        self.d_col_model = []
        if self.mode == 'classify':
            self.label = categories_to_int(self.ori_dataframe[args['label_name']].values.reshape(-1), -1, self.memory,
                                           self.isvalid)
            self.ori_dataframe = self.ori_dataframe.copy()
            self.ori_dataframe[args['label_name']] = self.label
        else:
            self.label = self.ori_dataframe[args['label_name']].values
        self.refresh_states()

    @property
    def get_memory(self):
        return self.memory

    def refresh_states(self):
        '''重置随着episode变化的属性'''
        # 随着step变化的属性，每个episode需要被重置
        logging.debug(f'Start discrete_fe_2_int_type')
        self.discrete_fe_2_int_type()
        self.continuous_combine = self.ori_dataframe[self.continuous_columns]
        # self.continuous_encode = pd.DataFrame()
        # self.continuous_temp = pd.DataFrame()

        self.ori_cols_continuous = self.ori_dataframe[self.continuous_columns]
        logging.debug(f'Start continous_fe_2_norm')
        # self.ori_c_columns_norm = self.continous_fe_2_norm()
        self.ori_c_columns_norm = self.ori_cols_continuous.copy()
        self.continuous_reward = self.ori_c_columns_norm.copy()
        self.continuous_encode = self.ori_c_columns_norm.copy()
        # self.c_fes_norm_out = None
        # self.c_fes_scale_out = None

    def __refresh_continuous_actions__(self, actions_c):
        '''每个step更新连续特征操作'''
        # print(actions_c)
        self.value_convert = actions_c['value_convert'] if 'value_convert' in actions_c else {}  # dict
        self.delete_c = actions_c['delete'] if 'delete' in actions_c else {}  # dict
        self.value_convert2 = actions_c['value_convert2'] if 'value_convert2' in actions_c else {}  # dict
        self.add_ = actions_c['add'] if 'add' in actions_c else []
        self.subtract_ = actions_c['subtract'] if 'subtract' in actions_c else []
        self.multiply_ = actions_c['multiply'] if 'multiply' in actions_c else []
        self.divide_ = actions_c['divide'] if 'divide' in actions_c else []
        self.selector_c = actions_c['selector_c'] if 'selector_c' in actions_c else {}

    def __refresh_discrete_actions__(self, actions_d):
        '''每个mdp step需运行该函数'''
        self.Cn2 = actions_d['two'] if 'two' in actions_d else []
        self.Cn3 = actions_d['three'] if 'three' in actions_d else []
        self.Cn4 = actions_d['four'] if 'four' in actions_d else []
        self.bins = actions_d['bins1'] if 'bins1' in actions_d else {}
        self.selector_d = actions_d['selector_d'] if 'selector_d' in actions_d else {}
        self.combine = actions_d["combine"] if "combine" in actions_d else []
        self.delete_d = actions_d["delete"] if "delete" in actions_d else {}
    def discrete_fe_2_int_type(self):
        '''
        用于将离散特征转换为整数类型
        '''
        all_names = self.ori_dataframe.columns.tolist()[:-1]
        self.discrete_encode = pd.DataFrame()
        self.discrete_combine = pd.DataFrame()
        self.discrete_reward = pd.DataFrame()

        for index, col in enumerate(all_names):
            if col in self.discrete_columns:
                ori_fe = self.ori_dataframe[col].values
                name = col + '_int'
                res = get_candidate_feature(name, self.isvalid)
                if res is not None:
                    int_type_col = res
                else:
                    int_type_col = categories_to_int(ori_fe, col, self.memory, self.isvalid)
                    if not self.isvalid and Candidate_features.shape[0] == len(int_type_col):
                        Candidate_features[col] = int_type_col
                self.discrete_reward[col] = int_type_col

                if len(np.unique(int_type_col)) > self.bins_thresh or self.isvalid:
                    name = col + '_bin_dis'
                    res = get_candidate_feature(name, self.isvalid)
                    if res is not None:
                        ori_fe_bins = res
                    else:
                        ori_fe_bins = binning_for_discrete(int_type_col, self.bins_thresh, self.mode, self.label, index,
                                                           self.memory, self.isvalid)
                        if not self.isvalid and Candidate_features.shape[0] == len(ori_fe_bins):
                            Candidate_features[col] = ori_fe_bins
                    self.discrete_combine[col] = ori_fe_bins
                    # self.discrete_encode[col] = normalization(ori_fe_bins).reshape(-1)
                else:
                    self.discrete_combine[col] = int_type_col
                    # self.discrete_encode[col] = normalization(int_type_col).reshape(-1)
            else:
                name = col + '_bin_tree'
                res = get_candidate_feature(name, self.isvalid)
                if res is not None:
                    c_fe_bins = res
                else:
                    c_fe_bins = binning_with_tree(self.ori_dataframe[col].values, self.label, col_index=index,
                                                  memory=self.memory, isvalid=self.isvalid)
                    if not self.isvalid and Candidate_features.shape[0] == len(c_fe_bins):
                        Candidate_features[col] = c_fe_bins
                self.discrete_combine[col] = c_fe_bins
                # self.discrete_encode[col] = normalization(c_fe_bins).reshape(-1)
                # self.discrete_encode = self.discrete_encode.values
                # self.discrete_reward = self.discrete_reward.values
                # self.discrete_combine = self.discrete_combine.values


    def continous_fe_2_norm(self):
        '''
        continuous_fe_2_norm(self): 该方法用于对连续型特征进行归一化处理。首先创建一个空的DataFrame fes_after_norm，然后遍历每个连续型特征列，并调用normalization()函数对每个特征进行归一化处理。将得到的归一化后的特征存储在fes_after_norm中并返回。
        '''
        fes_after_norm = pd.DataFrame()
        for col in self.continuous_columns:
            ori_col = self.ori_cols_continuous[col].values
            scaled_fe = normalization(ori_col, col, self.memory, self.isvalid)
            fes_after_norm[col] = scaled_fe.reshape(-1)
        return fes_after_norm

    def single_fe_operations(self):
        """
        single_fe_operations(self): 该方法用于对连续型特征进行单个特征操作。
        根据self.value_convert字典中存储的索引与操作的映射关系，遍历每个操作。
        如果操作为"None"，则跳过；
        如果操作为"terminate"，则从连续型特征数据集中删除相应特征列，并添加到self.terminate_list列表中；
        如果操作为"delete"，则从连续型特征数据集和归一化后的特征数据集中删除相应特征列；
        否则，在当前特征的基础上进行操作，生成新的特征并将其添加到连续型特征数据集和归一化后的特征数据集中。
        """
        for index, operation in self.value_convert.items():
        # 值转换与原始特征无关，当前特征做值转换。reward统一用归一化的值计算。
            if index in self.terminate_list or self.continuous_combine.shape[1] == 0:
                continue
            ori_col = self.continuous_combine.iloc[:, index].copy()
            # ori_col = self.continuous_reward.iloc[:, index].copy()
            if operation == 'None':
                # self.continuous_temp[ori_col.name] = ori_col
                # fe_norm = normalization(ori_col, ori_col.name, self.memory, self.isvalid)
                # self.continuous_encode[ori_col.name] = fe_norm.reshape(-1)
                continue
            elif operation == "terminate":
                self.continuous_combine = self.continuous_combine.drop(labels=[ori_col.name], axis=1)
                self.continuous_encode = self.continuous_encode.drop(labels=[ori_col.name], axis=1)
                self.terminate_list.append(index)
                continue
            elif operation == "delete":
                self.continuous_reward = self.continuous_reward.drop(labels=[ori_col.name], axis=1)
                self.continuous_combine = self.continuous_combine.drop(labels=[ori_col.name], axis=1)
                self.continuous_encode = self.continuous_encode.drop(labels=[ori_col.name], axis=1)
                continue
            else:
                name = ori_col.name + '_' + operation
                res = get_candidate_feature(name, self.isvalid)
                if res is not None:
                    new_fe = res.values
                else:
                    new_fe = globals()[operation](ori_col.values)
                    if not self.isvalid: Candidate_features[name] = new_fe
                    # self.continuous_combine[name] = new_fe.reshape(-1)
                    self.continuous_combine.loc[:, name] = new_fe.reshape(-1).copy()
                    # fe_norm = normalization(new_fe, name, self.memory, self.isvalid)
                    # self.continuous_encode[name] = fe_norm.reshape(-1)
                    # self.continuous_reward[name] = fe_norm.reshape(-1)
                    self.continuous_encode.loc[:, name] = new_fe.reshape(-1).copy()
                    self.continuous_reward.loc[:, name] = new_fe.reshape(-1).copy()
                    # self.continuous_encode.loc[:, name] = fe_norm.reshape(-1).copy()
                    # self.continuous_reward.loc[:, name] = fe_norm.reshape(-1).copy()


    def arithmetic_operations(self):
        """
        arithmetic_operations(self):
        该方法用于对连续型特征进行加法、减法、乘法和除法等算术操作。
        根据不同的操作（'add', 'subtract', 'multiply', 'divide'），
        遍历相应的特征组合列表combine_feature_tuples_list。
        对于每个特征组合，获取相应的列索引并从连续型特征数据集中取出相应的列。
        然后，根据操作执行相应的算术运算，并生成新的特征。
        将新的特征添加到连续型特征数据集、归一化后的特征数据集以及奖励数据集中。
        """
        operations = ['add', 'subtract', 'multiply', 'divide']
        feature_informations = [self.add_, self.subtract_, self.multiply_, self.divide_]

        for i,feature_information in enumerate(feature_informations):
            if len(feature_information) == 0:
                continue
            # combine_feature_tuples_list = combine_feature_tuples(feature_information, 2)
            combine_feature_tuples_list = feature_information
            operation = operations[i]
            for col_index_tuple in combine_feature_tuples_list:
                col1_index, col2_index = col_index_tuple
                if col2_index in self.terminate_list or self.continuous_combine.shape[1] == 0:
                    continue
                # col1_index表示当前特征的索引,col2_index表示原始特征的索引
                col1 = self.continuous_combine.iloc[:, col1_index]
                col2 = self.ori_cols_continuous.iloc[:, col2_index]
                if col1.equals(col2):
                    continue

                name = col1.name + '_' + operation + '_' + col2.name
                res1 = get_candidate_feature(name, self.isvalid)
                if res1 is not None:
                    new_fe = res1.values
                elif operation in ['add', 'multiply']:
                    name2 = col2.name + '_' + operation + '_' + col1.name
                    res2 = get_candidate_feature(name2, self.isvalid)
                    if res2 is not None:
                        new_fe = res2.values
                    else:
                        new_fe = globals()[operation](col1.values, col2.values)
                        if not self.isvalid:
                            Candidate_features[name] = new_fe
                else:
                    new_fe = globals()[operation](col1.values, col2.values)
                    if not self.isvalid:
                        Candidate_features[name] = new_fe

                self.continuous_combine = self.continuous_combine.copy()
                self.continuous_encode = self.continuous_encode.copy()
                self.continuous_reward = self.continuous_reward.copy()
                self.continuous_combine[name] = new_fe.reshape(-1)
                # fe_norm = normalization(new_fe, name, self.memory, self.isvalid)
                # self.continuous_encode[name] = fe_norm.reshape(-1)
                # self.continuous_reward[name] = fe_norm.reshape(-1)
                self.continuous_encode[name] = new_fe.reshape(-1)
                self.continuous_reward[name] = new_fe.reshape(-1)

    def binning_operation(self):
        '''
        该方法用于对特征进行分箱操作。
        首先获取所有特征的列名列表all_names，然后复制原始特征数据集ori_cols。
        接着遍历self.bins字典中的每个键值对，其中键表示特征索引，值表示分箱数。
        根据特征索引获取对应的列名和分箱数，并将原始特征数据取出来。
        如果原始特征的唯一值个数大于分箱数或者self.isvalid为True，则调用binning_for_discrete()函数对离散型特征进行分箱操作；
        否则，调用binning()函数对连续型特征进行分箱操作。
        分箱操作返回更新后的原始特征数据，并将其重新赋给对应的列。
        最后，更新self.ori_cols为更新后的原始特征数据集。
        '''
        all_names = self.ori_dataframe.columns.tolist()[:-1]
        ori_cols = self.ori_cols.copy()
        for index, bins in self.bins.items():
            col_name = all_names[index]
            bins = int(bins)
            ori_fe = ori_cols[:, index]
            if len(np.unique(ori_fe)) > bins or self.isvalid:
                if col_name in self.discrete_columns:
                    ori_fe = binning_for_discrete(ori_fe, bins, self.mode, self.label,index, self.memory, self.isvalid)
                else:
                    ori_fe, fre_list = binning(ori_fe, bins, index,memory=self.memory, isvalid=self.isvalid)
                    ori_fe = ori_fe.reshape(-1)
                ori_cols[:, index] = ori_fe
        self.ori_cols = ori_cols

    def select_d_features(self):
        """
        select_d_features(self): 该方法用于根据强化学习代理的选择对离散型特征进行筛选。
        首先复制原始特征数据集ori_cols，然后创建一个全为1的掩码数组ori_mask，
        长度与特征数量相同。接着遍历self.selector_d字典中的每个键值对，其中键表示特征索引，值表示是否被选择。
        根据特征索引将对应的掩码值更新为相应的值。
        最后，根据掩码数组选择被选中的特征索引，并将其对应的列取出来，更新self.ori_cols为更新后的原始特征数据集。
        :return:
        """
        ori_cols = self.ori_cols.copy()
        ori_mask = np.ones(ori_cols.shape[1])
        for index, mask in self.selector_d.items():
            ori_mask[index] = int(mask)
        selected_index = np.argwhere(ori_mask == 1).reshape(-1)
        ori_cols = ori_cols[:, selected_index]
        self.ori_cols = ori_cols

    def select_c_features(self,c_cols):
        """
        select_c_features(self, c_cols): 该方法用于根据强化学习代理的选择对连续型特征进行筛选。
        接收连续型特征数据集作为参数c_cols，
        然后创建一个全为1的掩码数组ori_mask，长度与特征数量相同。
        接着遍历self.selector_c字典中的每个键值对，其中键表示特征索引，值表示是否被选择。
        根据特征索引将对应的掩码值更新为相应的值。
        最后，根据掩码数组选择被选中的特征索引，并从连续型特征数据集中取出对应的列，并返回所选特征的数据集
        :param c_cols:
        :return:
        """
        ori_mask = np.ones(c_cols.shape[1])
        for index, mask in self.selector_c.items():
            ori_mask[index] = int(mask)
        selected_index = np.argwhere(ori_mask == 1).reshape(-1)
        return c_cols[:, selected_index]

    # def delete_features(self):
    #     '''select continuous features due to RL agent'''
    #     index = list(self.delete_c.keys())
    #     name_delete = self.continuous_combine.iloc[:, index].columns
    #     self.continuous_reward.drop(labels=name_delete, axis=1, inplace=True)

    def feature_cross_operations(self,ori_fes=None):
        """
        feature_cross_operations(self, ori_fes=None): 该方法用于对离散型特征进行特征交叉操作。
        根据预定义的操作类型列表和对应的特征信息列表，依次进行特征交叉操作。
        如果操作类型不为"None"，则根据特征信息生成要进行特征交叉的特征组合列表，并逐个处理每个特征组合。
        首先根据特征索引列表从原始特征数据集ori_cols中取出相应的列，
        然后调用features_combine_ori()函数对这些特征进行特征交叉操作，返回新生成的特征。
        如果ori_fes是一个NumPy数组，则将新生成的特征水平堆叠到ori_fes上；
        否则，直接赋值给ori_fes。
        最后，如果ori_fes是一个NumPy数组，则将其与self.ori_cols水平堆叠起来更新self.ori_cols。
        :param ori_fes:
        :return:
        """
        operations = ['Cn2', 'Cn3', 'Cn4']
        feature_informations = [self.Cn2, self.Cn3, self.Cn4]
        for i, feature_information in enumerate(feature_informations):
            operation = operations[i]
            if operation != 'None':
                type = int(operation[-1])
                combine_feature_tuples_list = combine_feature_tuples(feature_information, type)
                for combine_feature_tuple in combine_feature_tuples_list:
                    combine_feature_index_list = list(combine_feature_tuple)
                    ori_cols = self.ori_cols[:, combine_feature_index_list]
                    # print(combine_feature_index_list)
                    new_fe = features_combine_ori(ori_cols, combine_feature_index_list,self.memory, self.isvalid)
                    if isinstance(ori_fes, np.ndarray):
                        ori_fes = np.hstack((ori_fes, new_fe))
                    else:
                        ori_fes = new_fe
        if isinstance(ori_fes, np.ndarray):
            self.ori_cols = np.hstack((self.ori_cols, ori_fes))


    def feature_combine(self):
        """
        feature_combine(self): 该方法用于对离散型特征进行特征合并操作。
        遍历self.combine列表中的每个元素，其中每个元素包含了两个索引：index和action。
        根据这两个索引从离散特征组合数据集中取出对应的列，并生成新特征的名称。
        然后检查是否已经存在该特征的候选特征，如果存在，则使用其值作为新特征；
        否则，根据可行值字典和离散特征的组合生成新特征。
        最后，将新特征添加到离散特征组合数据集和奖励数据集中，并更新self.discrete_reward和self.discrete_combine。
        :return:
        """
        for index, action in self.combine:
            ori_fe1 = self.discrete_combine.iloc[:, index]
            ori_fe2 = self.discrete_combine.iloc[:, action]
            name = ori_fe1.name + '_combine_' + ori_fe2.name
            res1 = get_candidate_feature(name, self.isvalid)
            if res1 is not None:
                new_fe = res1.values
            else:
                name2 = ori_fe2.name + '_combine_' + ori_fe1.name
                res2 = get_candidate_feature(name2, self.isvalid)
                if res2 is not None:
                    new_fe = res2.values
                else:
                    feasible_values = {}
                    cnt = 0
                    for x in np.unique(ori_fe1):
                        for y in np.unique(ori_fe2):
                            feasible_values[str(int(x)) + str(int(y))] = cnt
                            cnt += 1
                    new_fe = generate_combine_fe(self.discrete_combine.iloc[:, [index, action]].values, feasible_values)

                    if not self.isvalid: Candidate_features[name] = new_fe

            self.discrete_combine[name] = new_fe.reshape(-1)
            if self.discrete_reward.shape[0]:
                self.discrete_reward[name] = new_fe.reshape(-1)
            else:
                self.discrete_reward = new_fe.reshape(-1)
            # self.discrete_encode[name] = normalization(new_fe).reshape(-1)
            # self.discrete_combine = np.hstack((self.discrete_combine, new_fe))
            # if self.discrete_reward.shape[0]:
            #     self.discrete_reward = np.hstack((self.discrete_reward, new_fe))
            # else:
            #     self.discrete_reward = new_fe
            # self.discrete_encode = np.hstack((self.discrete_encode, normalization(new_fe)))


    def process_discrete(self, actions):
        '''处理所有离散变量'''
        for action in actions:
            self.__refresh_discrete_actions__(action)
            # 四步，分箱和combine
            # self.binning_operation()  # 第一次分箱
            # self.feature_cross_operations()  # feature_combine
            # self.select_d_features()
            self.feature_combine()
        return self.discrete_combine, self.discrete_reward
        # return self.discrete_encode, self.discrete_combine
        # return self.discrete_combine, self.discrete_combine

    def process_continuous(self, actions):
        '''处理所有连续变量'''
        for action in actions:
            self.__refresh_continuous_actions__(action)
            self.arithmetic_operations()
            self.single_fe_operations()
        return self.continuous_encode, self.continuous_reward
        # return self.continuous_encode, self.continuous_combine
        # return self.continuous_combine, self.continuous_combine











