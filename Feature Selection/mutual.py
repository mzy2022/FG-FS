import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

from utils import restore_parameters, make_onehot, ConcreteSelector
from copy import deepcopy
from tqdm import tqdm


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

class DynamicSelection(nn.Module):
    def __init__(self, VAE,attention,selector, predictor, mask_layer):
        super().__init__()

        # Set up models and mask layer.
        self.vae = VAE
        self.selector = selector
        self.predictor = predictor
        self.mask_layer = mask_layer
        self.attention = attention

        # Set up selector layer.
        self.selector_layer = ConcreteSelector()

    def fit(self,train_loader,val_loader,lr,nepochs,max_features,loss_fn,val_loss_fn=None,val_loss_mode=None,
            factor=0.2,patience=2,min_lr=1e-5,early_stopping_epochs=None,start_temp=1.0,end_temp=0.1,
            temp_steps=5,argmax=False,verbose=True,alpha=1.0,beta=0.2):

        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = 'min'
        else:
            if val_loss_mode is None:
                raise ValueError('must specify val_loss_mode (min or max) when validation_loss_fn is specified')
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1

        # Set up models.
        selector = self.selector
        predictor = self.predictor
        mask_layer = self.mask_layer
        selector_layer = self.selector_layer
        attention = self.attention
        device = next(predictor.parameters()).device
        val_loss_fn.to(device)

        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, y = next(iter(val_loader))
            assert len(x.shape) == 2
            mask_size = x.shape[1]

        # For tracking best models with zero temperature.
        best_val = None
        best_zerotemp_selector = None
        best_zerotemp_predictor = None

        # Train separately with each temperature.
        total_epochs = 0

        for temp in np.geomspace(start_temp, end_temp, temp_steps):
            if verbose:
                print(f'Starting training with temp = {temp:.4f}\n')

            # Set up optimizer and lr scheduler.
            opt = optim.Adam(set(list(predictor.parameters()) + list(selector.parameters()) + list(attention.parameters())), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=val_loss_mode, factor=factor, patience=patience,
                min_lr=min_lr, verbose=verbose)

            # For tracking best models and early stopping.
            best_selector = deepcopy(selector)
            best_predictor = deepcopy(predictor)
            num_bad_epochs = 0

            for epoch in tqdm(range(nepochs), desc='Training', unit='epoch'):
                selector.train()
                predictor.train()
                attention.train()

                for x, y in train_loader:
                    # Move to device.
                    x = x.to(device)
                    y = y.to(device)

                    m = torch.zeros(len(x), mask_size, dtype=x.dtype, device=device)
                    selector.zero_grad()
                    predictor.zero_grad()
                    attention.zero_grad()
                    # vae.zero_grad()

                    for _ in range(max_features):
                        x_masked = mask_layer(x, m)
                        # x_attention = attention(x_masked)
                        # x_masked = x_attention
                        logits = selector(x_masked).flatten(1)
                        soft = selector_layer(logits, temp, deterministic=False)
                        m_soft = torch.max(m, soft)
                        x_masked = mask_layer(x, m_soft)
                        pred = predictor(x_masked)
                        loss = loss_fn(pred, y)

                        #######KL散度###############################
                        kl_past = self.vae.impute(x,m)
                        kl_now = self.vae.impute(x,m_soft)
                        kl_past_dims = kl_past.shape[1] // 2
                        kl_past_mean = kl_past[:, :kl_past_dims]
                        kl_past_std = torch.exp(kl_past[:, kl_past_dims:])

                        kl_now_dims = kl_now.shape[1] // 2
                        kl_now_mean = kl_now[:, :kl_now_dims]
                        kl_now_std = torch.exp(kl_now[:, kl_now_dims:])
                        kl = torch.distributions.kl_divergence(
                            Normal(kl_past_mean, kl_past_std),
                            Normal(kl_now_mean, kl_now_std)).sum(1)
                        kl = kl.sum()
                        ##########################################kl散度###########
                        final_loss = alpha * (loss / max_features) - beta * (kl / max_features)
                        final_loss.backward()
                        # (loss / max_features).backward()


                        # Update mask, ensure no repeats.
                        m = torch.max(m, make_onehot(selector_layer(logits - 1e6 * m, 1e-6)))

                    opt.step()

                # Calculate validation loss.
                selector.eval()
                predictor.eval()
                attention.eval()
                with torch.no_grad():
                    # For mean loss.
                    pred_list = []
                    hard_pred_list = []
                    label_list = []
                    for x, y in val_loader:
                        # Move to device.
                        x = x.to(device)
                        y = y.to(device)

                        m = torch.zeros(len(x), mask_size, dtype=x.dtype, device=device)
                        for _ in range(max_features):
                            # Evaluate selector model.
                            x_masked = mask_layer(x, m)
                            # x_attention = attention(x_masked)
                            # x_masked = x_attention
                            logits = selector(x_masked).flatten(1)

                            # Get selections, ensure no repeats.
                            logits = logits - 1e6 * m
                            if argmax:
                                soft = selector_layer(logits, temp, deterministic=True)
                            else:
                                soft = selector_layer(logits, temp)
                            m_soft = torch.max(m, soft)
                            # For calculating temp = 0 loss.
                            m = torch.max(m, make_onehot(soft))

                            # Evaluate predictor with soft sample.
                            x_masked = mask_layer(x, m_soft)
                            pred = predictor(x_masked)

                            # Evaluate predictor with hard sample.
                            x_masked = mask_layer(x, m)
                            hard_pred = predictor(x_masked)

                            # Append predictions and labels.
                            pred_list.append(pred.cpu())
                            hard_pred_list.append(hard_pred.cpu())
                            label_list.append(y.cpu())

                    # Calculate mean loss.
                    pred = torch.cat(pred_list, 0)
                    hard_pred = torch.cat(hard_pred_list, 0)
                    y = torch.cat(label_list, 0)
                    val_loss = val_loss_fn(pred, y)
                    val_hard_loss = val_loss_fn(hard_pred, y)

                if verbose:
                    print(f'{"-" * 8}Epoch {epoch + 1} ({epoch + 1 + total_epochs} total){"-" * 8}')
                    print(f'Val loss = {val_loss:.4f}, Zero-temp loss = {val_hard_loss:.4f}\n')

                # Update scheduler.
                scheduler.step(val_loss)

                # Check if best model.
                if val_loss == scheduler.best:
                    best_selector = deepcopy(selector)
                    best_predictor = deepcopy(predictor)
                    num_bad_epochs = 0
                else:
                    num_bad_epochs += 1

                # Check if best model with zero temperature.
                if ((best_val is None)
                        or (val_loss_mode == 'min' and val_hard_loss < best_val)
                        or (val_loss_mode == 'max' and val_hard_loss > best_val)):
                    best_val = val_hard_loss
                    best_zerotemp_selector = deepcopy(selector)
                    best_zerotemp_predictor = deepcopy(predictor)

                # Early stopping.
                if num_bad_epochs > early_stopping_epochs:
                    break

            # Update total epoch count.
            if verbose:
                print(f'Stopping temp = {temp:.4f} at epoch {epoch + 1}\n')
            total_epochs += (epoch + 1)

            # Copy parameters from best model.
            restore_parameters(selector, best_selector)
            restore_parameters(predictor, best_predictor)

        # Copy parameters from best model with zero temperature.
        restore_parameters(selector, best_zerotemp_selector)
        restore_parameters(predictor, best_zerotemp_predictor)

    def forward(self, x, max_features, argmax=True):
        '''
        Make predictions using selected features.

        Args:
          x:
          max_features:
          argmax:
        '''
        # Setup.
        selector = self.selector
        predictor = self.predictor
        mask_layer = self.mask_layer
        attention = self.attention
        selector_layer = self.selector_layer
        device = next(predictor.parameters()).device

        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = self.mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        m = torch.zeros(len(x), mask_size, device=device)

        for _ in range(max_features):
            # Evaluate selector model.
            x_masked = mask_layer(x, m)
            # x_attention = attention(x_masked)
            # x_masked = x_attention
            logits = selector(x_masked).flatten(1)

            # Update selections, ensure no repeats.
            logits = logits - 1e6 * m
            if argmax:
                m = torch.max(m, make_onehot(logits))
            else:
                m = torch.max(m, make_onehot(selector_layer(logits, 1e-6)))



        # Make predictions.
        x_masked = mask_layer(x, m)
        pred = predictor(x_masked)
        return pred, x_masked, m

    def evaluate(self,
                 loader,
                 max_features,
                 metric,
                 argmax=True):
        '''
        Evaluate mean performance across a dataset.

        Args:
          loader:
          max_features:
          metric:
          argmax:
        '''
        # Setup.
        self.selector.eval()
        self.predictor.eval()
        self.attention.eval()
        device = next(self.predictor.parameters()).device

        # For calculating mean loss.
        pred_list = []
        label_list = []

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device)

                # Calculate loss.
                pred, x_masked, m = self.forward(x, max_features, argmax)
                pred_list.append(pred.cpu())
                label_list.append(y.cpu())

            # Calculate metric(s).
            y = torch.cat(label_list, 0)
            pred = torch.cat(pred_list, 0)
            if isinstance(metric, (tuple, list)):
                score = [m(pred, y).item() for m in metric]
            elif isinstance(metric, dict):
                score = {name: m(pred, y).item() for name, m in metric.items()}
            else:
                score = metric(pred, y).item()

        return score











