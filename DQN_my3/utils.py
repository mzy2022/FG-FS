import logging
import os
import sys


def log_dir(exp_dir):
    """Make directory of log file and initialize logging module"""
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log_format = '%(asctime)s %(filename)s:%(lineno)d %(funcName)s %(levelname)s | %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, force=True)
    fh = logging.FileHandler(os.path.join(exp_dir, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def get_key_from_dict(d, value):
    k = [k for k, v in d.items() if v == value]
    return k
