import random

from MLFE import Buffer, load, ENV, Model
from utils import *
from args import args
import os
import pandas as pd
from tqdm import tqdm

opt_type = args.opt_type
opt_size = 9 if opt_type == 'o1' else 5
qsa_size = args.qsa_size
history_size = args.history_size
input_size = qsa_size * 2 + opt_size * history_size + 1 * history_size + opt_size * 4 + 4
buffer_size = args.buffer_size
seed = args.seed
depth = args.depth
budget = args.budget
gamestep = args.gamestep
num_epochs = args.num_epochs
num_local_epochs = args.num_local_epochs
num_episodes = args.num_episodes
optimisation_steps = args.num_optimize_steps
n_jobs = args.n_jobs
tau = args.tau
gamma = args.gamma
epsilon = args.epsilon
batch_size = args.batch_size

num_process = args.multiprocessing
multiprocessing = True if args.multiprocessing > 0 else False

save_model = True
# train = True
# test = True

out_dir = os.path.join(args.out_dir, 'cafem')
model_dir = os.path.join(args.out_dir, 'cafem_model')
if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)


class Tasks():
    def __init__(self, datasetids, buffer_size=100):
        self.datasets = {}
        self.tasks = []
        if os.path.isfile('../data/log.csv'):
            log = pd.read_csv('../data/log.csv', header=None)
            for val in log.values:
                self.tasks.append([val[0], val[1], Buffer(buffer_size)])
        else:
            f = open('../data/log.csv', 'a')
            for did in tqdm(datasetids):
                f_dataset = "../data/%d/%d.arff" % (did, did)
                dataset, meta, tasktype = load(f_path=f_dataset)
                self.datasets[did] = (dataset, meta)
                for i, v in enumerate(meta[:-1]):
                    if v == "numeric":
                        f.write("%d,%d\n" % (did, i))
                        self.tasks.append([did, i, Buffer(buffer_size)])
            f.close()

    def sample(self, n):
        tasks = random.sample(self.tasks, n)
        return tasks


def generate_trajectories(task):
    f_dataset = "../data/%d/%d.arff" % (task[0], task[0])
    weights = task[2]
    dataset, meta,tasktype = load(f_path=f_dataset)
    env = ENV(dataset, feature=task[1], maxdepth=depth, evalcount=budget,
              opt_type=opt_type, random_state=seed, n_jobs=1)
    tmp_buffer = []
    # print('env done')
    localmodel = Model(opt_size=opt_size, input_size=input_size, name="model", maml=False)
    # print('model done')