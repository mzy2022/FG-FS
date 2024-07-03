import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import ray
from ray import tune

import inquirer
from Own.ppo_attention.emedding_data.gated_tab_transformer import GatedTabTransformer
from Own.ppo_attention.emedding_data.train_with_validation import train, validate
from Own.ppo_attention.emedding_data.data_utils import get_unique_categorical_counts, get_categ_cont_target_values, train_val_test_split
from Own.ppo_attention.emedding_data.metadata import datasets

# device = torch.device("cpu")
# LOG_INTERVAL = 10
# MAX_EPOCHS = 200
# RANDOMIZE_SAMPLES = False
# DATASET = "german"
#
# data = datasets[DATASET]
# dataset = pd.read_csv(data["PATH"], header=0)
#
# if RANDOMIZE_SAMPLES:
#     # Randomize order
#     dataset = dataset.sample(frac=1)
#
# n_categories = get_unique_categorical_counts(dataset, data["CONT_COUNT"])
#
# train_dataframe = dataset
#
# train_cont, train_categ, train_target = get_categ_cont_target_values(train_dataframe, data["POSITIVE_CLASS"], data["CONT_COUNT"])

def train_experiment(config,n_categories,train_cont, train_categ, train_target):
    model = GatedTabTransformer(
        categories = n_categories,                          # tuple containing the number of unique values within each category
        num_continuous = train_cont.shape[1],               # number of continuous values
        transformer_dim = config["transformer_dim"],        # dimension, paper set at 32
        dim_out = 1,                                        # binary prediction, but could be anything
        transformer_depth = config["transformer_depth"],    # depth, paper recommended 6
        transformer_heads = config["transformer_heads"],    # heads, paper recommends 8
        attn_dropout = config["dropout"],                   # post-attention dropout
        ff_dropout = config["dropout"],                     # feed forward dropout
        mlp_act = nn.LeakyReLU(config["relu_slope"]),       # activation for final mlp, defaults to relu, but could be anything else (selu, etc.)
        mlp_depth=config["mlp_depth"],                      # mlp hidden layers depth
        mlp_dimension=config["mlp_dimension"],              # dimension of mlp layers
        gmlp_enabled=config["gmlp_enabled"]                 # gmlp or standard mlp
    )

    model = model.train().to(device=device)

    new_x_d,new_x_c = train(
        model,
        train_cont,
        train_categ,
        train_target,
        device=device,
        batch_size=config["batch_size"],
        max_epochs=MAX_EPOCHS,
        patience=config["patience"],
        save_best_model_dict=False,
        save_metrics=False,
        log_interval=LOG_INTERVAL
    )

    return new_x_d,new_x_c

# new_x_d,new_x_c = train_experiment({
#         "batch_size": 64,
#         "patience": 5,
#         "initial_lr": 1e-3,
#         "scheduler_gamma": 0.1,
#         "scheduler_step": 8,
#         "relu_slope": 0,
#         "transformer_heads": 8,
#         "transformer_depth": 6,
#         "transformer_dim": 8,
#         "gmlp_enabled": True,
#         "mlp_depth": 6,
#         "mlp_dimension": 64,
#         "dropout": 0.2,
#     },n_categories=n_categories,train_cont=train_cont,train_categ=train_categ,train_target=train_target)
#
# print(new_x_d.shape,new_x_c.shape)