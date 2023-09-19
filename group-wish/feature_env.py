from collections import namedtuple
from utils.logger import error, info
from utils.tools import feature_state_generation, downstream_task_new, test_task_new, cluster_features

TASK_DICT = {'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary': 'cls',
             'bike_share': 'reg', 'german_credit': 'cls', 'higgs': 'cls',
             'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
             'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
             'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
             'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
             'Openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg'
             }

MEASUREMENT = {
    'cls': ['precision', 'recall', 'f1_score'],
    'reg': ['mae', 'mse', 'rae']
}

REPLAY = {
    'random', 'per'
}


class FeatureEnv:
    def __init__(self, task_name, task_type=None, ablation_mode=''):
        if task_type is None:
            self.task_type = TASK_DICT[task_name]
        else:
            self.task_type = task_type
        self.model_performance = namedtuple('ModelPerformance', MEASUREMENT[self.task_type])
        if ablation_mode == '-c':
            self.mode = 'c'
        else:
            self.mode = ''

    '''
            input a Dataframe (cluster or feature set)
            :return the feature status
            return type is Numpy array
    '''

    # 函数的目的是生成特征状态矩阵。它首先将输入 X 转换为浮点型（np.float64），然后使用describe()方法计算统计描述信息。
    def get_feature_state(self, data):
        return feature_state_generation(data)

    '''
            input a Dataframe (cluster or feature set)
            :return the current dataframe performance
            return type is Numpy array
    '''

    # 它用于执行下游任务（即基于数据进行分类或回归预测）并计算评估指标
    def get_reward(self, data):
        return downstream_task_new(data, self.task_type)

    '''
            input a Dataframe (cluster or feature set)
            :return the current dataframe performance on few dataset
            its related measure is listed in {MEASUREMENT[self.task_type]}
            return type is Numpy array
        '''

    def get_performance(self, data):
        a, b, c = test_task_new(data, task=self.task_type)
        return self.model_performance(a, b, c)

    # 用于将特征进行聚类，并返回聚类结果
    def cluster_build(self, X, y, cluster_num):
        return cluster_features(X, y, cluster_num, mode=self.mode)


# 这是一个名为report_performance的方法，用于报告模型在原始数据和优化后数据上的性能度量。
    def report_performance(self, original, opt):
        report = self.get_performance(opt)
        original_report = self.get_performance(original)
        if self.task_type == 'reg':
            final_result = report.rae
            info('MAE on original is: {:.3f}, MAE on generated is: {:.3f}'.format(original_report.mae, report.mae))
            info('RMSE on original is: {:.3f}, RMSE on generated is: {:.3f}'.format(original_report.mse, report.mse))
            info('1-RAE on original is: {:.3f}, 1-RAE on generated is: {:.3f}'.format(original_report.rae, report.rae))
        elif self.task_type == 'cls':
            final_result = report.f1_score
            info('Pre on original is: {:.3f}, Pre on generated is: {:.3f}'.format(original_report.precision,
                                                                                  report.precision))
            info('Rec on original is: {:.3f}, Rec on generated is: {:.3f}'.format(original_report.recall, report.recall))
            info(
                'F-1 on original is: {:.3f}, F-1 on generated is: {:.3f}'.format(original_report.f1_score, report.f1_score))
        elif self.task_type == 'det':
            final_result = report.ras
            info('Average Precision Score on original is: {:.3f}, Average Precision Score on generated is: {:.3f}'.format(
                original_report.map, report.map))
            info('F1 Score on original is: {:.3f}, F1 Score on generated is: {:.3f}'.format(original_report.f1_score,
                                                                                            report.f1_score))
            info('ROC AUC Score on original is: {:.3f}, ROC AUC Score on generated is: {:.3f}'.format(original_report.ras,
                                                                                                      report.ras))
        else:
            error('wrong task name!!!!!')
            assert False
        return final_result
