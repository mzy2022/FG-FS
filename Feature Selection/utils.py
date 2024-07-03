import os
import torch
import numpy as np
import pandas as pd
from torch.distributions import RelaxedOneHotCategorical
from torch.utils.data import TensorDataset, Subset
import torch.nn as nn

def load_data(data_path,features=None):
    # Load data.
    data = pd.read_csv(data_path)
    label_name = data.columns[-1]
    # data = pd.read_csv("spam.csv")

    # Set features.
    if features is None:
        features = np.array([f for f in data.columns if f not in [label_name]])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)

    # Extract x, y.
    x = np.array(data.drop([label_name], axis=1)[features]).astype('float32')
    y = np.array(data[label_name]).astype('int64')

    # Create dataset object.
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset


def data_split(dataset, val_portion=0.2, test_portion=0.2, random_state=0):
    '''
    Split dataset into train, val, test.

    Args:
      dataset: PyTorch dataset object.
      val_portion: percentage of samples for validation.
      test_portion: percentage of samples for testing.
      random_state: random seed.
    '''
    # Shuffle sample indices.
    rng = np.random.default_rng(random_state)
    inds = np.arange(len(dataset))
    rng.shuffle(inds)

    # Assign indices to splits.
    n_val = int(val_portion * len(dataset))
    n_test = int(test_portion * len(dataset))
    test_inds = inds[:n_test]
    val_inds = inds[n_test:(n_test + n_val)]
    train_inds = inds[(n_test + n_val):]

    # Create split datasets.
    test_dataset = Subset(dataset, test_inds)
    val_dataset = Subset(dataset, val_inds)
    train_dataset = Subset(dataset, train_inds)
    return train_dataset, val_dataset, test_dataset


def make_onehot(x):
    '''Make an approximately one-hot vector one-hot.'''
    argmax = torch.argmax(x, dim=1)
    onehot = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    onehot[torch.arange(len(x)), argmax] = 1
    # wwww = onehot.detach().cpu().numpy()
    return onehot

def restore_parameters(model, best_model):
    '''Move parameters from best model to current model.'''
    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param

def generate_uniform_mask(batch_size, num_features):
    '''Generate binary masks with cardinality chosen uniformly at random.'''
    unif = torch.rand(batch_size, num_features)
    ref = torch.rand(batch_size, 1)
    return (unif > ref).float()


class ConcreteSelector(nn.Module):
    '''Output layer for selector models.'''

    def __init__(self, gamma=0.2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, temp, deterministic=False):
        if deterministic:
            return torch.softmax(logits / (self.gamma * temp), dim=-1)
        else:
            dist = RelaxedOneHotCategorical(temp, logits=logits / self.gamma)
            return dist.rsample()


class MaskLayer(nn.Module):
    '''
    Mask layer for tabular data.

    Args:
      append:
      mask_size:
    '''

    def __init__(self, append, mask_size=None):
        super().__init__()
        self.append = append
        self.mask_size = mask_size

    def forward(self, x, m):
        out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out


class MaskLayerGrouped(nn.Module):
    '''
    Mask layer for tabular data with feature grouping.

    Args:
      group_matrix:
      append:
    '''

    def __init__(self, group_matrix, append):
        # Verify group matrix.
        assert torch.all(group_matrix.sum(dim=0) == 1)
        assert torch.all((group_matrix == 0) | (group_matrix == 1))

        # Initialize.
        super().__init__()
        self.register_buffer('group_matrix', group_matrix.float())
        self.append = append
        self.mask_size = len(group_matrix)

    def forward(self, x, m):
        out = x * (m @ self.group_matrix)
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out
