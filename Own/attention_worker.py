class unary_Worker(object):
    def __init__(self, args):
        self.args = args

        # self.continuous_col = args.continuous_col
        # self.discrete_col = args.discrete_col
        self.hp_action = None
        self.score_list = []
        self.process_data = None
        self.cluster_dict = None
        self.score = None
        self.dataframe = None
        self.emb_data_u = None
        self.states_u = None
        self.actions_u = None
        self.action_indexs_u = None
        self.prob_vecs_u = None
        self.action_entropys_u = None
        self.ori_states_u = None
        self.cluster_dict_u = None

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
    def __init__(self, args):
        self.args = args
        self.actions_prob = []
        self.hp_action = None
        self.score_list = []
        self.hp_action = None
        self.score_list = []
        self.process_data = None
        self.cluster_dict = None
        self.score = None
        self.dataframe = None
        self.emb_data_b = None
        self.states_b = None
        self.actions_b = None
        self.action_indexs_b = None
        self.prob_vecs_b = None
        self.action_entropys_b = None
        self.ori_states_b = None
        self.cluster_dict_b = None



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

