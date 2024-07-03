#!/bin/bash

export PYTHONPATH=$PYTHONPATH:D:\python files\pythonProject3
dataset=$"spectf"
cuda=$1
python ./autolearn/feat_selection/nfo/iter_train.py --eval_model=RF --data=$dataset --cuda=$cuda --hyper_config=default --feat_pool=$HOME/nfo/iter/$dataset --ckp_path=$HOME/nfo/ckps/$dataset