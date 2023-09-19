import logging
import os
import sys
import time

import numpy as np

def get_key_from_dict(d,value):
    k = [k for k,v in d.items() if v==value]
    return k

