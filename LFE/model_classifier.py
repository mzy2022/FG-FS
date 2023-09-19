import pandas as pd
import numpy as np
# import h5py
import math

import torch
from sklearn import ensemble, preprocessing, multiclass
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split

from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch.nn as nn
from torch import optim


# Transformation
def sqrt(col):
    return list(map(np.sqrt, col))


def freq(col):
    col = np.floor(col)
    counter = Counter(col)
    return [counter.get(elem) for elem in col]


def tanh(col):
    return list(map(np.tanh, col));


def log(col):
    return list(map(np.log, col));


def my_sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid(col):
    return list(map(my_sigmoid, col))


def square(col):
    return list(map(np.square, col))


def normalize_val(num, col_min, col_max, norm_range):
    width = col_max - col_min

    if (width == 0): width = 1

    return (num - col_min) / width * (norm_range[1] - norm_range[0]) + norm_range[0]


def normalize(col):
    norm_range = (-1, 1)
    col_max = np.amax(col)
    col_min = np.amin(col)

    return list(map(lambda x: normalize_val(x, col_min, col_max, norm_range), col))


dids = np.load("magicindexes.npy")
seed = 67
transformations = [sqrt, freq, tanh, log, sigmoid, square, normalize]
transformations_name = ["sqrt", "freq", "tanh", "log", "sigmoid", "square", "normalize"]
trans2target1 = {}
trans2target2 = {}
trans2target3 = {}

# Comrpessed Dataset paramters
qsa_representation = []
num_bin = 10
too_big = 10000

# Neural Nets Parameters and Variables
MLP_LFE_Nets = {}
inp_shape = (2, num_bin)
dropout = 0.2
norm = (0, 10)
pred_threshold = 0.51
train_set_max = 80000
test_set_max = 80000


# def binarize_dataset():
def load_dataset(id):
    X = np.load("datasets/binary_numeric/" + str(id) + "-data.npy")
    y = np.load("datasets/binary_numeric/" + str(id) + "-target.npy")
    categorical = np.load("datasets/binary_numeric/" + str(id) + "-categorical.npy")
    return X, y, categorical


def evaluate_model(X, y, categorical):
    imp = SimpleImputer(missing_values=np.nan)
    X = imp.fit_transform(X)
    enc = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical)])
    X = enc.fit_transform(X)
    clf = ensemble.RandomForestClassifier(random_state=seed)
    return cross_val_score(clf, X, y, cv=10)


def is_positive(X, y, categorical, base_score, transformation, feature):
    transformed_feature = np.array(transformation(X[:, feature]))
    X = np.c_[X, transformed_feature]
    categorical = np.append(categorical, False)
    new_score = evaluate_model(X, y, categorical).mean()
    return 1 if (base_score <= (new_score - 0.01)) else 0


def is_positive_2(X, y, categorical, base_score, transformation, feature):
    transformed_feature = np.array(transformation(X[:, feature]))
    new_score = evaluate_model(transformed_feature.reshape(-1, 1), y, [False]).mean()

    return 1 if (base_score <= (new_score - 0.005)) else 0


def is_positive_3(X, y, categorical, base_score, transformation, feature):
    transformed_feature = np.array(transformation(X[:, feature]))
    new_score = evaluate_model(transformed_feature.reshape(-1, 1), y, [False]).mean()

    return 1 if (new_score > base_score * 1.01) else 0


# Build the target for the compressed feature
bad_dataset = []


def build_target_for_compressed(dids, bad_datasets=None):
    if bad_datasets is None:
        bad_datasets = []
    for transf in transformations:
        trans2target1[transf] = []
        trans2target2[transf] = []
        trans2target3[transf] = []
    for did in dids:
        print("Start dataset number", did)
        try:
            X, y, categorical = load_dataset(did)
            new_indexes = []
            if X.shape[0] > too_big:
                new_indexes = np.random.choice(X.shape[0], too_big, replace=False)
                X = X[new_indexes]
                y = y[new_indexes]

            base_score = evaluate_model(X, y, categorical).mean()

            # Find the indexes of numeric attributes
            numerical_indexes = np.where(np.invert(categorical))[0]
            sample_numerical_indexes = np.random.choice(numerical_indexes, min(numerical_indexes.shape[0], 10),
                                                        replace=False)

            for i, transf in enumerate(transformations):
                for feature in sample_numerical_indexes:
                    print("\tEvaluating feature " + str(feature))
                    mlp_target_1 = is_positive(X, y, categorical, base_score, transf, feature)
                    mlp_target_2 = is_positive_2(X, y, categorical, base_score, transf, feature)
                    mlp_target_3 = is_positive_3(X, y, categorical, base_score, transf, feature)
                    print("\t\t" + str(mlp_target_1), str(mlp_target_2), str(mlp_target_3))

                    trans2target1[transf].append((did, feature, mlp_target_1))
                    trans2target2[transf].append((did, feature, mlp_target_2))
                    trans2target3[transf].append((did, feature, mlp_target_3))

        except:
            print("The evaluation of dataset " + str(did) + " failed")
            bad_datasets.append(did)
            continue


def save_target_for_compressed(path):
    for transf, name in zip(transformations, transformations_name):
        np.save(path + name + "1", trans2target1[transf])
        np.save(path + name + "2", trans2target2[transf])
        np.save(path + name + "3", trans2target3[transf])


def normalize_Rx(matrix):
    Rxc = np.zeros(shape=matrix.shape)
    for i, row in enumerate(matrix):
        max_c = np.amax(row)
        min_c = np.amin(row)
        bin_width = (max_c - min_c) / (norm[1] - norm[0])
        Rxc[i] = np.apply_along_axis(lambda x: np.floor((x - min_c) / bin_width + norm[0]), 0, row)
    return Rxc

def to_quantile_sketch_array(did, col, targets, bins, t_class, index):
    max_c = np.nanmax(col)
    min_c = np.nanmin(col)
    bin_width = (max_c-min_c)/num_bin
    Rx = np.zeros(shape=(2,num_bin))

    if bin_width == 0:
        return

    for val,y in zip(col,targets):
        if not np.isnan(val):
            bin_value = int(np.floor((val - min_c) / bin_width))
            bin_value = np.clip(bin_value, 0, num_bin - 1)
            my_class = 0 if t_class == y else 1
            Rx[my_class][bin_value] = Rx[my_class][bin_value] + 1

        Rx = normalize_Rx(Rx)

        qsa_representation.append(np.insert(Rx.flatten(), 0, [did, index]))


def build_compressed_dataset(dids):
    for did in dids:
        print("Start dataset number", did)
        try:
            X, y, categorical = load_dataset(did)
            if X.shape[0] > too_big:
                new_indexes = np.random.choice(X.shape[0], too_big, replace=False)
                X = X[new_indexes]
                y = y[new_indexes]

            numerical_indexes = np.where(np.invert(categorical))[0]
            classes = set(y)

            for t_class in classes:
                for index in numerical_indexes:
                    to_quantile_sketch_array(did, X[:, index], y, num_bin, t_class, index)
        except:
            print("Error with dataset " + str(did))
            continue

# Save the compressed datasets
def save_compressed_dataset(path):
    np.save(path + "compressed.npy", qsa_representation)


# CREATING THE NEURAL NETS
class MLP(nn.Module):
    def __init__(self, inp_shape):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(inp_shape, 64)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 64)
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(64, 1)
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.flatten(out)
        out = self.fc3(out)
        out = self.relu2(out)
        return out


# 创建并存储每个变换对应的模型
for transf in transformations_name:
    model = MLP(10)
    MLP_LFE_Nets[transf] = model


def load_compressed_ds():
    data = pd.DataFrame(np.load("datasets/compressed/compressed.npy"))
    data.columns = [str(i) for i in range (num_bin*2+2)]
    return data

def assemble_training_set(compressed, transformation_targets):
    targetDf = pd.DataFrame(transformation_targets)
    targetDf.columns = ["0", "1", "2"]
    merged = pd.merge(compressed, targetDf, how='left', on=["0", "1"])
    class_1 = merged.iloc[:, 2:num_bin + 2].values
    class_2 = merged.iloc[:, num_bin + 2:-1].values
    target = np.array(merged.iloc[:, -1].values)
    meta_inf = np.array(merged.iloc[:, :2].values)
    meta_target = np.c_[target, meta_inf]

    X = np.concatenate((class_1[:, None], class_2[:, None]), axis=1)

    return X, meta_target

def split_training_test():
    compressed_ds = load_compressed_ds()
    for transf, name in zip(transformations, transformations_name):
        transformation_targets = np.load("datasets/compressed/" + name + "3.npy")
        X, y = assemble_training_set(compressed_ds, transformation_targets)
        X_s_tr, X_s_test, y_s_tr, y_s_test = train_test_split(X, y, test_size=0.3)

        # Dropping the meta-info from training set
        y_s_tr = y_s_tr[:, :1]

        np.save("datasets/training/" + name + "-data_split", X_s_tr)
        np.save("datasets/training/" + name + "-target_split", y_s_tr)
        np.save("datasets/test/" + name + "-data_split", X_s_test)
        np.save("datasets/test/" + name + "-target_split", y_s_test)


def load_training_set(transf):
    X = np.load("datasets/training/" + transf + "-data_split.npy")
    y = np.load("datasets/training/" + transf + "-target_split.npy")

    return X, y



def balance_dataset(X, y, pos_perc = 0.5):
    X = np.array(X)
    y = np.array(y)

    cnt = Counter(y)

    neg_num = cnt[0]
    pos_num = cnt[1]

    neg_index = (y == 0)
    pos_index = (y == 1)

    X_pos = X[pos_index]
    X_neg = X[neg_index]
    y_pos = y[pos_index]
    y_neg = y[neg_index]

    if pos_perc >= 0.5:
        if pos_num <= neg_num:
            sample_neg_num = int(X_pos.shape[0] * (1/pos_perc - 1))
            neg_sampled_indexes = np.random.choice(X_neg.shape[0], sample_neg_num, replace=False)
            X_neg = X_neg[neg_sampled_indexes]
            y_neg = y_neg[neg_sampled_indexes]

        else:
            sample_neg_ideal_size = int(X_pos.shape[0] * (1/pos_perc - 1))

            if sample_neg_ideal_size > X_neg.shape[0]:
                sample_pos_num = int(X_neg.shape[0] * (1/(1-pos_perc) - 1))
                pos_sampled_indexes = np.random.choice(X_pos.shape[0], sample_pos_num, replace=False)
                X_pos = X_pos[pos_sampled_indexes]
                y_pos = y_pos[pos_sampled_indexes]
            else:
                neg_sampled_indexes = np.random.choice(X_neg.shape[0], sample_neg_ideal_size, replace=False)
                X_neg = X_neg[neg_sampled_indexes]
                y_neg = y_neg[neg_sampled_indexes]
    else:
        if pos_num <= neg_num:
            sample_pos_ideal_size = int(X_neg.shape[0] * (1/(1-pos_perc) - 1))

            if(sample_pos_ideal_size > X_pos.shape[0]):
                sample_neg_num = int(X_pos.shape[0] * (1/pos_perc - 1))
                neg_sampled_indexes = np.random.choice(X_neg.shape[0], sample_neg_num, replace=False)
                X_neg = X_neg[neg_sampled_indexes]
                y_neg = y_neg[neg_sampled_indexes]
            else:
                pos_sampled_indexes = np.random.choice(X_pos.shape[0], sample_pos_ideal_size, replace=False)
                X_pos = X_pos[pos_sampled_indexes]
                y_pos = y_pos[pos_sampled_indexes]
        else:
            sample_pos_num = int(X_neg.shape[0] * (1/(1-pos_perc) - 1))
            pos_sampled_indexes = np.random.choice(X_pos.shape[0], sample_pos_num, replace=False)
            X_pos = X_pos[pos_sampled_indexes]
            y_pos = y_pos[pos_sampled_indexes]

    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.concatenate((y_pos, y_neg) , axis=0)

    shuffle_index = np.random.choice(X.shape[0], X.shape[0], replace=False)

    X = X[shuffle_index]
    y = y[shuffle_index]

    return X,y


def train_MLPs():
    for transf, name in zip(transformations, transformations_name):
        print("Start training " + name)

        X, y = load_training_set(name)

        # if(X.shape[0] > train_set_max):
        #    new_indexes = np.random.choice(X.shape[0], too_big, replace=False)
        #    X = X[new_indexes]
        #    y = y[new_indexes]

        X, y = balance_dataset(X, y.reshape(y.shape[0]), pos_perc=0.5)
        print("dataset_balanced")

        y = y.reshape(y.shape[0], 1)

        model = MLP_LFE_Nets[name]()
        print(model)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        num_epochs = 500
        batch_size = 32

        dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for inputs, targets in dataloader:
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataset)

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))

        # Save the model
        torch.save(model.state_dict(), "path/to/save/model.pt")


def save_MLPs():
    for transf in transformations_name:
        torch.save(MLP_LFE_Nets[transf].state_dict(), "datasets/MLPs/" + transf + "-weights")
        torch.save(MLP_LFE_Nets[transf], "datasets/MLPs/" + transf + "-net_model")


def load_MLPs():
    for name in transformations_name:
        loaded_model = torch.load("datasets/MLPs/" + name + "-net_model")
        loaded_model.load_state_dict(torch.load("datasets/MLPs/" + name + "-weights"))

        # Set model to evaluation mode
        loaded_model.eval()

        MLP_LFE_Nets[name] = loaded_model


def load_test_set():
    X = np.load("datasets/test/" + transformations_name[0] + "-data_split.npy")
    y_meta = np.load("datasets/test/" + transformations_name[0] + "-target_split.npy")
    t = np.full((y_meta.shape[0], 1), 0)
    y_meta = np.concatenate((y_meta, t), axis=1)

    for i, name in enumerate(transformations_name[1:]):
        X = np.concatenate((X, np.load("datasets/test/" + name + "-data_split.npy")), axis=0)
        y_meta_tmp = np.load("datasets/test/" + name + "-target_split.npy")
        t = np.full((y_meta_tmp.shape[0], 1), i + 1)
        y_meta_tmp = np.concatenate((y_meta_tmp, t), axis=1)
        y_meta = np.concatenate((y_meta, y_meta_tmp), axis=0)

    return X, y_meta


def evaluate_transformation_classifier():
    # Number of prediction on features
    num_of_prediction = {}
    # Number of correct prediction on features
    num_of_correct_prediction = {}
    # Number of dataset which received a prediction
    good_predicted_dids = set()
    num_of_predicted_dataset = 0

    pred_mat = []

    X, y_meta = load_test_set()

    if (X.shape[0] > test_set_max):
        new_indexes = np.random.choice(X.shape[0], too_big, replace=False)
        X = X[new_indexes]
        y_meta = y_meta[new_indexes]

    for transf in transformations_name:
        pred_mat.append(MLP_LFE_Nets[transf].predict(X))
        num_of_prediction[transf] = 0
        num_of_correct_prediction[transf] = 0

    pred_mat = np.array(pred_mat).transpose()

    for predictions, did, feature in zip(pred_mat[0], y_meta[:, 1], y_meta[:, 2]):
        pmax = np.amax(predictions)
        print(predictions)

        if pmax > pred_threshold:
            index = np.where(predictions == pmax)[0][0]
            print(index)
            num_of_prediction[transformations_name[index]] += 1

            # Select the target for the transformation and the dataset
            positive_example_found = np.where((y_meta[:, 0] == 1) & \
                                              (y_meta[:, 1] == did) & \
                                              (y_meta[:, 2] == feature) & \
                                              (y_meta[:, 3] == index)) \
                                         [0].shape[0] > 0

            if (positive_example_found):
                print("found!")
                good_predicted_dids.add(did)
                num_of_correct_prediction[transformations_name[index]] += 1

    global_correct_pred = 0;
    global_pred = 0

    for transf in transformations_name:

        if (num_of_prediction[transf] == 0):
            print("No predictions have been made")
            continue

        global_pred += num_of_prediction[transf];
        global_correct_pred += num_of_correct_prediction[transf]

        print("Evalutation of the transformation classifier: " + transf)
        print("\tNumber of prediction:", num_of_prediction[transf])
        print("\tNumber of Correct prediciton:", num_of_correct_prediction[transf])
        print("Accuracy:", num_of_correct_prediction[transf] / num_of_prediction[transf])

    print("\n")
    print("Number of datasets who received a good prediction:", len(good_predicted_dids))
    print("Total number of positive examples: " + str(np.where(y_meta[:, 0] == 1)[0].shape[0]))
    print("Total number of examples: " + str(X.shape[0]))
    print("Total accuracy: " + str(global_correct_pred / global_pred))
    print("Total recall: " + str((global_correct_pred) / (np.where(y_meta[:, 0] == 1)[0].shape[0])))


def main():
    build_target_for_compressed(dids)
    save_target_for_compressed("datasets/compressed/")
    build_compressed_dataset(dids)
    save_compressed_dataset("datasets/compressed/")
    split_training_test()
    initialize_MLPs()
    train_MLPs()
    save_MLPs()
    load_MLPs()
    evaluate_transformation_classifier()


