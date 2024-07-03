import copy
import torch
import uuid
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def validate(model, cont_data, categ_data, target_data, device="cuda", val_batch_size=1, save_metrics=True):
    model = model.eval()
    results = np.zeros((categ_data.shape[0], 1))

    for i in range(categ_data.shape[0] // val_batch_size):
        x_categ = torch.tensor(categ_data[val_batch_size*i:val_batch_size*i+val_batch_size]).to(dtype=torch.int64, device=device)
        x_cont = torch.tensor(cont_data[val_batch_size*i:val_batch_size*i+val_batch_size]).to(dtype=torch.float32, device=device)

        pred = model(x_categ, x_cont)
        results[val_batch_size*i:val_batch_size*i+val_batch_size, 0] = torch.sigmoid(pred).squeeze().cpu().detach().numpy()

    fpr, tpr, _ = metrics.roc_curve(target_data[:results.shape[0]], results[:, 0])
    if save_metrics:
        fig, ax = plt.subplots(1, 1)
        plt.xlim([0,1])
        plt.ylim([0,1])
        ax.plot(fpr, tpr)
        plt.savefig(f'{uuid.uuid4()}.png')

    area = metrics.auc(fpr, tpr)
    model = model.train()
    return area


def train(
    model,
    train_cont,
    train_categ,
    train_target,
    device="cpu",
    batch_size=64,
    max_epochs=100,
    patience=10,
    save_best_model_dict=True,
    save_metrics=True,
    log_interval=10
):
        x_categ = torch.tensor(train_categ).to(dtype=torch.int64, device=device)
        x_cont = torch.tensor(train_cont).to(dtype=torch.float32,device=device)
        y_target = torch.tensor(train_target).to(dtype=torch.float32,device=device)
        pred = model(x_categ, x_cont)
        return pred