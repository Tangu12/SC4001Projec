import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

def get_ptbdb_dataset(noise = []):
    classes = 2

    df_normal = pd.read_csv("ECG_heartbeats/ptbdb/ptbdb_normal.csv", header=None)
    df_abnormal = pd.read_csv("ECG_heartbeats/ptbdb/ptbdb_abnormal.csv", header=None)
    # print(df_normal.shape)
    # print(df_abnormal.shape)
    df = pd.concat([df_normal, df_abnormal], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # I don't really und what is gg on here :P
    X = df.iloc[:, :-1].values.astype(np.float32)  # selecting every column except the last column
    y = df.iloc[:, -1].values.astype(np.int64)     # just selecting the last column

    # Augmentation
    # X = fn_aug(X)

    # Standardize per-feature (helps training)
    # m, s = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True) + 1e-8
    # X = (X - m) / s

    # Reshape to (N, C, L) for Conv1d: here C=1, L=187
    X = X[:, None, :]  # (N, 1, 187)

    X_t = torch.from_numpy(X)  # creates a CPU tensor that shares the same underlying memory as X
    y_t = torch.from_numpy(y)

    # Train/val split (80/20)
    full_ds = TensorDataset(X_t, y_t)
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, drop_last=False)
    test_dl = 0

    return train_dl, val_dl, test_dl, classes

def get_mitbih_dataset(noise = []):
    classes = 5

    df_train = pd.read_csv("ECG_heartbeats/mitbih/mitbih_train.csv", header=None)
    df_test = pd.read_csv("ECG_heartbeats/mitbih/mitbih_test.csv", header=None)

    train_x, train_y = df_train.iloc[:, :-1].values.astype(np.float32), df_train.iloc[:, -1].values.astype(np.int64)
    test_x, test_y = df_test.iloc[:, :-1].values.astype(np.float32), df_test.iloc[:, -1].values.astype(np.int64)

    train_x = train_x[:, None, :]
    test_x = test_x[:, None, :]

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, drop_last=False)
    test_dl = 0

    return train_dl, val_dl, test_dl, classes