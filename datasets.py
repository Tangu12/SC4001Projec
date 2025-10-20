import pandas as pd
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from data_augmentation import *
from enum import Enum
from typing import Optional, List

from sklearn.utils import shuffle

class data_aug(Enum):
    gaussian_noise = 1
    salt_pepper = 2
    amplitude_drift = 3
    signal_shift = 4

class data_eng(Enum):
    min_max = 1
    mean_std = 2
    skew = 3
    kurtosis = 4

def data_manipulation(df, d_aug = None, d_eng = None):
    df = df.copy()

    if d_aug is not None or d_eng is not None:

        # Get the valid lengths of each input, used for both data augmentation and data engineering
        _, valid_lens = get_valid_lengths(df)

        # Only if d_aug is given
        if d_aug is not None:

            # Data augmentation (Currently applying every augmentation no exclusiveness)
            if data_aug.gaussian_noise in d_aug:
                df = add_gaussian_noise(df, valid_lens)

            if data_aug.salt_pepper in d_aug:
                df = add_salt_pepper(df, valid_lens)

            if data_aug.amplitude_drift in d_aug:
                df = add_amplitude_drift(df, valid_lens)

            if data_aug.signal_shift in d_aug:
                df = add_signal_shift(df, valid_lens)

        # Only if d_eng is given
        if d_eng is not None:

            # Data engineering
            if data_eng.min_max in d_eng:
                df = add_min_max(df, valid_lens)

            if data_eng.mean_std in d_eng:
                df = add_mean_std(df, valid_lens)

            if data_eng.skew in d_eng:
                df = add_skewness(df, valid_lens)

            if data_eng.kurtosis in d_eng:
                df = add_kurtosis(df, valid_lens)

    return df

def get_ptbdb_dataset(d_aug: Optional[List[data_aug]] = None, d_eng: Optional[List[data_eng]] = None):
    classes = 2

    df_normal = pd.read_csv("ECG_heartbeats/ptbdb/ptbdb_normal.csv", header=None)
    df_abnormal = pd.read_csv("ECG_heartbeats/ptbdb/ptbdb_abnormal.csv", header=None)
    df = pd.concat([df_normal, df_abnormal], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True) # Combining df and shuffling

    df_1 = data_manipulation(df, d_eng = d_eng) # Original data append with data engineering
    if d_aug is not None:
        df_2 = data_manipulation(df, d_aug, d_eng) # Added noise and append with data engineering
        df = pd.concat([df_1, df_2], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # print(df)

    n_total = len(df)
    n_train = int(0.8 * n_total)

    df_train = df.to_numpy()[:n_train].copy()
    df_test = df.to_numpy()[n_train:].copy()

    # preprocess_pipeline = Pipeline([
    #     ('scaler', StandardScaler()),
    # ])
    #
    # x_train, y_train = preprocess_pipeline.fit_transform(df_train[:,:-1]), df_train[:,-1]
    # x_test, y_test = preprocess_pipeline.transform(df_test[:,:-1]), df_test[:,-1]

    x_train, y_train = df_train[:,:-1].astype(np.float32),  df_train[:,-1].astype(np.int64)
    x_test, y_test = df_test[:,:-1].astype(np.float32), df_test[:,-1].astype(np.int64)

    # x_train = x_train[:, None, :]  # (N, 1, 187)
    # x_test = x_test[:, None, :]

    # x_train_tensor = torch.from_numpy(x_train).float()
    # y_train_tensor = torch.from_numpy(y_train).long()
    # x_test_tensor = torch.from_numpy(x_test).float()
    # y_test_tensor = torch.from_numpy(y_test).long()
    #
    # train_ds = TensorDataset(x_train_tensor, y_train_tensor)
    # test_ds = TensorDataset(x_test_tensor, y_test_tensor)

    return classes, x_train, y_train, x_test, y_test

def get_mitbih_dataset(d_aug: Optional[List[data_aug]] = None, d_eng: Optional[List[data_eng]] = None):
    classes = 5

    df_train = pd.read_csv("ECG_heartbeats/mitbih/mitbih_train.csv", header=None)
    df_test = pd.read_csv("ECG_heartbeats/mitbih/mitbih_test.csv", header=None)

    df_train_1 = data_manipulation(df_train, d_eng)
    df_test = data_manipulation(df_test, d_eng)

    if d_aug is not None:
        df_train_2 = data_manipulation(df_train, d_aug, d_eng)
        df_train = pd.concat([df_train_1, df_train_2], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

    df_train = df_train.to_numpy().copy()
    df_test = df_test.to_numpy().copy()

    # preprocess_pipeline = Pipeline([
    #     ('scaler', StandardScaler()),
    # ])
    #
    # x_train, y_train = preprocess_pipeline.fit_transform(df_train[:,:-1]), df_train[:,-1]
    # x_test, y_test = preprocess_pipeline.transform(df_test[:,:-1]), df_test[:,-1]

    x_train, y_train = df_train[:,:-1].astype(np.float32),  df_train[:,-1].astype(np.int64)
    x_test, y_test = df_test[:,:-1].astype(np.float32), df_test[:,-1].astype(np.int64)

    # x_train = x_train[:, None, :]  # (N, 1, 187)
    # x_test = x_test[:, None, :]

    # x_train_tensor = torch.from_numpy(x_train).float()
    # y_train_tensor = torch.from_numpy(y_train).long()
    # x_test_tensor = torch.from_numpy(x_test).float()
    # y_test_tensor = torch.from_numpy(y_test).long()
    #
    # train_ds = TensorDataset(x_train_tensor, y_train_tensor)
    # test_ds = TensorDataset(x_test_tensor, y_test_tensor)

    return classes, x_train, y_train, x_test, y_test