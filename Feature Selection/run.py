import os
import argparse
import numpy as np
import torch.nn as nn
import tqdm as tqdm
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AUROC
from utils import load_data, data_split,MaskLayer
from mutual import DynamicSelection
from pvae import PVAE
import sys
from self_attention import Attention
from pretrainer import MaskingPretrainer

sys.path.append('../')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='syn1',
                    choices=['spam', 'diabetes', 'miniboone'])
parser.add_argument('--method', type=str, default='eddi',
                    choices=['sage', 'permutation', 'deeplift', 'intgrad',
                             'cae', 'iterative', 'eddi', 'greedy'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_trials', type=int, default=1)
parser.add_argument('--num_restarts', type=int, default=1)
parser.add_argument('--dataset_path', type=str, default='datasets/syn1.csv',
                    choices=['datasets/spam.csv', 'datasets/diabetes.csv', 'datasets/miniboone.csv'])

# Various configurations.
num_features_dict = {
    'spam': list(range(1, 11)) + list(range(12, 55,5)),
    'diabetes': list(range(1, 44)),
    'miniboone': list(range(1, 11)) + list(range(15, 30, 5)),
    'syn1': list(range(1, 20))
}
max_features_dict = {
    'spam': 35,
    'diabetes': 35,
    'miniboone': 35,
    'syn1':15,
}


def get_network(d_in, d_out):
    hidden = 128
    dropout = 0.3
    model = nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_out))
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.dataset_path
    num_features = num_features_dict[args.dataset]
    device = torch.device('cuda', args.gpu)
    dataset = load_data(data_path)
    d_in = dataset.input_size
    d_out = dataset.output_size

    mean = dataset.tensors[0].mean(dim=0)
    std = torch.clamp(dataset.tensors[0].std(dim=0), min=1e-3)
    dataset.tensors = ((dataset.tensors[0] - mean) / std, dataset.tensors[1])
    train_dataset, val_dataset, test_dataset = data_split(dataset)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, pin_memory=True)

    for trial in range(args.num_trials):
        # For saving results.
        results_dict = {
            'auroc': {},
            'acc': {},
            'features': {}
        }
        auroc_metric = lambda pred, y: AUROC(task='multiclass', num_classes=d_out)(pred.softmax(dim=1), y)
        acc_metric = Accuracy(task='multiclass', num_classes=d_out)



    # Train PVAE.
    bottleneck = 16
    encoder = get_network(d_in * 2, bottleneck * 2)
    decoder = get_network(bottleneck, d_in)
    mask_layer = MaskLayer(append=True)
    pv = PVAE(encoder, decoder, mask_layer, 128, 'gaussian').to(device)
    pv.fit(
        train_loader,
        val_loader,
        lr=1e-3,
        nepochs=250,
        verbose=False)

    ##self_attention
    data_nums = d_in * 2
    d_model = 128
    d_k = 32
    d_v = 32
    d_ff = 64
    n_heads = 6
    attention = Attention(data_nums,d_model,d_k,d_v,d_ff,n_heads)
    ### 主框架
    predictor = get_network(d_in * 2, d_out)
    selector = get_network(d_in * 2, d_in)
    mask_layer = MaskLayer(append=True)

    pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
    pretrain.fit(
        train_loader,
        val_loader,
        lr=1e-3,
        nepochs=250,
        loss_fn=nn.CrossEntropyLoss(),
        patience=5,
        verbose=False)

    gdfs = DynamicSelection(pv,attention,selector, predictor, mask_layer).to(device)
    gdfs.fit(
        train_loader,
        val_loader,
        lr=1e-3,
        nepochs=250,
        max_features=max_features_dict[args.dataset],
        loss_fn=nn.CrossEntropyLoss(),
        patience=5,
        verbose=True)

    for num in num_features:
        auroc, acc = gdfs.evaluate(test_loader, num, (auroc_metric, acc_metric))
        results_dict['auroc'][num] = auroc
        results_dict['acc'][num] = acc
        print(f'Num = {num}, AUROC = {100 * auroc:.2f}, Acc = {100 * acc:.2f}')
    avg_auroc = 0
    avg_acc = 0
    for i in list(results_dict['auroc'].values())[0:10]:
        avg_auroc += i
    print(avg_auroc / 10)
    for i in list(results_dict['acc'].values())[0:10]:
        avg_acc += i
    print(avg_acc / 10)




