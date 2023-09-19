import os
import sys
import torch
import warnings
BASE_DIR = os.path.dirname(os.path.abspath('.'))
sys.path.append(BASE_DIR)

torch.manual_seed(0)
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
os.environ['DATA_ABS_PATH'] = BASE_DIR + '/data/processed'

import argparse

def init_param():
    parser = argparse.ArgumentParser(description='PyTorch Experiment')
    parser.add_argument('--name', type=str, default='wine_red',help='data name')
    args, _ = parser.parse_known_args()
    return args