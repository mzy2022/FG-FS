class Worker(object):
    def __init__(self, args):
        self.args = args
        self.states = None
        self.actions_otp = None
        self.actions_ops = None
        self.values = None
        self.accs = None
        self.model = None
        self.orders = None
        self.con_or_dis = None
        self.reward_1 = None
        self.reward_2 = None
        self.special = None
        self.best_score = 0
        self.new_x = None
        self.adj_matrix = None
        self.input_x = None
        self.input_y = None
        self.inner_actions = None
        self.inner_log_prob = None
        self.inner_action_softmax = None
        self.inner_reward = None
        self.inner_actions_ = None
        self.inner_log_prob_ = None
        self.inner_action_softmax_ = None
        self.reward_ = None
        self.select_feature_nums = None

