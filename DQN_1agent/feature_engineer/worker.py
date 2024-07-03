class Worker(object):
    def __init__(self, args):
        self.ars = args
        self.states = None
        self.actions_otp = None
        self.actions_ops = None
        self.values = None
        self.accs = None
        self.model = None
        self.orders = None
        self.steps = None
        self.features_c = None
        self.features_d = None
        self.df = None
        self.scores = None
        self.scores_test = None
        self.state = None
        self.actions_ops = None
        self.actions_otp = None
        self.ff = None
        self.encodes_states = None
        self.states_ = None
        self.encodes_states_ = None
        self.actions_ops_ = None
        self.actions_otp_ = None
        self.features_c_ = None
        self.features_d_ = None
        self.ff_ = None
        self.states_ops = None
        self.states_otp = None
        self.states_otp_ = None
        self.states_ops_ = None
        self.c_d = None             #经过ops和otp的dataframe
        self.reward = None
        self.scores_b = None


