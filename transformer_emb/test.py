import numpy as np
import pandas as pd

import modules
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero

# %%

device = torch.device('cpu')
# Docs: https://yura52.github.io/delu/0.0.4/reference/api/zero.improve_reproducibility.html
zero.improve_reproducibility(seed=123456)

dataset = sklearn.datasets.fetch_california_housing()
task_type = 'regression'

# dataset = sklearn.datasets.fetch_covtype()
# task_type = 'multiclass'

assert task_type in ['binclass', 'multiclass', 'regression']

# X_all = dataset['data'].astype('float32')
X_all = pd.read_csv('spectf.csv')
X_all = X_all.iloc[:,:-1].values.astype('float32')
# y_all = dataset['target'].astype('float32' if task_type == 'regression' else 'int64')
# if task_type != 'regression':
#     y_all = sklearn.preprocessing.LabelEncoder().fit_transform(y_all).astype('int64')
# n_classes = int(max(y_all)) + 1 if task_type == 'multiclass' else None
#
# X = {}
# y = {}
# X['train'], X['test'], y['train'], y['test'] = sklearn.model_selection.train_test_split(
#     X_all, y_all, train_size=0.8
# )
# X['train'], X['val'], y['train'], y['val'] = sklearn.model_selection.train_test_split(
#     X['train'], y['train'], train_size=0.8
# )
#
# # not the best way to preprocess features, but enough for the demonstration
# preprocess = sklearn.preprocessing.StandardScaler().fit(X['train'])
# X = {
#     k: torch.tensor(preprocess.transform(v), device=device)
#     for k, v in X.items()
# }
# y = {k: torch.tensor(v, device=device) for k, v in y.items()}
#
# # !!! CRUCIAL for neural networks when solving regression problems !!!
# if task_type == 'regression':
#     y_mean = y['train'].mean().item()
#     y_std = y['train'].std().item()
#     y = {k: (v - y_mean) / y_std for k, v in y.items()}
# else:
#     y_std = y_mean = None
#
# if task_type != 'multiclass':
#     y = {k: v.float() for k, v in y.items()}

# d_out = 1


model = modules.FTTransformer.make_default(
    n_samples=X_all.shape[0],
    n_num_features=X_all.shape[1],
    cat_cardinalities=None,
    last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
    d_out=1,
)


model.to(device)
# optimizer = (
#     model.make_default_optimizer()
#     if isinstance(model, modules.FTTransformer)
#     else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# )
# loss_fn = (
#     F.binary_cross_entropy_with_logits
#     if task_type == 'binclass'
#     else F.cross_entropy
#     if task_type == 'multiclass'
#     else F.mse_loss
# )


# def apply_model(x_num, x_cat=None):
#     if isinstance(model, modules.FTTransformer):
#         return model(x_num, x_cat)
#     else:
#         raise NotImplementedError(
#             f'Looks like you are using a custom model: {type(model)}.'
#             ' Then you have to implement this branch first.'
#         )


all_data = torch.tensor(X_all)
mmmm = model(all_data,x_cat=None)



print(mmmm)
# @torch.no_grad()
# def evaluate(part):
#     model.eval()
#     prediction = []
#     for batch in zero.iter_batches(X[part], 1024):
#         prediction.append(apply_model(batch))
#     prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
#     target = y[part].cpu().numpy()
#
#     if task_type == 'binclass':
#         prediction = np.round(scipy.special.expit(prediction))
#         score = sklearn.metrics.accuracy_score(target, prediction)
#     elif task_type == 'multiclass':
#         prediction = prediction.argmax(1)
#         score = sklearn.metrics.accuracy_score(target, prediction)
#     else:
#         assert task_type == 'regression'
#         score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std
#     return score
#
#
# # Create a dataloader for batches of indices
# # Docs: https://yura52.github.io/delu/reference/api/zero.data.IndexLoader.html
# batch_size = 256
# train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)
#
# # Create a progress tracker for early stopping
# # Docs: https://yura52.github.io/delu/reference/api/zero.ProgressTracker.html
# progress = zero.ProgressTracker(patience=100)
#
# print(f'Test score before training: {evaluate("test"):.4f}')
#
# all_data = torch.tensor(X_all)
# mmmm = apply_model(all_data)
#
# n_epochs = 1000
# report_frequency = len(X['train']) // batch_size // 5
# for epoch in range(1, n_epochs + 1):
#     for iteration, batch_idx in enumerate(train_loader):
#         model.train()
#         optimizer.zero_grad()
#         x_batch = X['train'][batch_idx]
#         y_batch = y['train'][batch_idx]
#         mmmm = apply_model(x_batch)
#         loss = loss_fn(mmmm.squeeze(1), y_batch)
#         loss.backward()
#         optimizer.step()
#         if iteration % report_frequency == 0:
#             print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')
#
#     val_score = evaluate('val')
#     test_score = evaluate('test')
#     print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
#     progress.update((-1 if task_type == 'regression' else 1) * val_score)
#     if progress.success:
#         print(' <<< BEST VALIDATION EPOCH', end='')
#     print()
#     if progress.fail:
#         break
