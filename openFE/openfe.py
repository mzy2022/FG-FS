import gc
import os
import warnings
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from FeatureGenerator import *
import random
from concurrent.futures import ProcessPoolExecutor
import traceback
from utils import tree_to_formula, check_xor, formula_to_tree
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score
import scipy.special
from copy import deepcopy
from tqdm import tqdm
# import tracemalloc
from datetime import datetime
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

