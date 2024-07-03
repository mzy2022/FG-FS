
class unary_Worker(object):
    def __init__(self, args, history,is_uanry):
        self.args = args
        self.history = history
        # self.continuous_col = args.continuous_col
        # self.discrete_col = args.discrete_col
        self.unary_fe_history = history['unary_feature_engineering']
        self.actions_trans = self.get_actions_trans(self.unary_fe_history,is_unary=is_uanry)
        self.actions_prob = self.get_prob_vec(self.unary_fe_history)
        self.hp_action = None
        self.score_list = []
        self.process_data = None
        self.cluster_dict = None
        self.score = None



    @staticmethod
    def get_actions_trans(history,is_unary):
        if is_unary:
            actions_trans = {}
            for num, samples in history:
                if num not in actions_trans:
                    actions_trans[num] = []
                for sample in samples:
                    action = sample['unary_actions_name']
                    actions_trans[num].append(action)
        return actions_trans

    @staticmethod
    def get_prob_vec(history):
        actions_prob = {}
        for num, samples in history:
            if num not in actions_prob:
                actions_prob[num] = []
            for sample in samples:
                action = sample['prob_vec']
                actions_prob[num].append(action)

        return actions_prob


class binary_Worker(object):
    def __init__(self, args, history,is_uanry):
        self.args = args
        self.history = history
        # self.continuous_col = args.continuous_col
        # self.discrete_col = args.discrete_col
        self.binary_fe_history = history['binary_feature_engineering']
        self.actions_trans = self.get_actions_trans(self.binary_fe_history,is_unary=is_uanry)
        self.actions_prob = []
        self.actions_prob = self.get_prob_vec(self.binary_fe_history)
        # self.actions_prob = [sample[1]['prob_vec'] for sample in self.binary_fe_history]
        self.hp_action = None
        self.score_list = []



    @staticmethod
    def get_actions_trans(history,is_unary):
        if not is_unary:
            actions_trans = {}
            for num,samples in history:
                if num not in actions_trans:
                    actions_trans[num] = []
                for sample in samples:
                    action = sample['binary_actions_name']
                    actions_trans[num].append(action)
        return actions_trans

    @staticmethod
    def get_prob_vec(history):
        actions_prob = {}
        for num, samples in history:
            if num not in actions_prob:
                actions_prob[num] = []
            for sample in samples:
                action = sample['prob_vec']
                actions_prob[num].append(action)

        return actions_prob

