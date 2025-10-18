import numpy as np
import pandas as pd

def mask_padded(values, length):
    return values[:length]

# Get the valid length for each sequence
def get_valid_lengths(df, pad_value=0.0):
    features = df.iloc[:, :-1].copy()
    values = features.to_numpy()

    valid_lengths = []
    for row in values:
        # Find indices where the value != pad_value (=0)
        non_pad_idx = np.where(row != pad_value)[0]
        if len(non_pad_idx) == 0:
            valid_lengths.append(0)
        else:
            valid_lengths.append(non_pad_idx[-1] + 1)

    return features, np.array(valid_lengths)

# At each time stamp, add random noise using error ~ N(0,sigma)
def add_gaussian_noise(df, valid_lengths, sigma=0.01):
    features = df.iloc[:, :-1].copy()
    labels = df.iloc[:, -1]

    noisy = features.to_numpy().copy()
    for i, length in enumerate(valid_lengths):
        noise = np.random.normal(0, sigma, length)
        noisy[i, :length] += noise
        # Keep padded zeros untouched beyond valid length

    noisy = pd.DataFrame(noisy, columns=features.columns)
    return pd.concat([noisy, labels], axis=1)

# At each time stamp, have a probability of being replaced by min/max value of that sample
def add_salt_pepper(df, valid_lengths, prob=0.01):
    features = df.iloc[:, :-1].copy()
    labels = df.iloc[:, -1]

    noisy = features.to_numpy().copy()
    for i, length in enumerate(valid_lengths):
        signal = features.iloc[i, :length].to_numpy()
        mask = np.random.rand(length) < prob
        min_val, max_val = signal.min(), signal.max()

        # Randomly choose between min or max
        replacement = np.random.choice([min_val, max_val], size=length)
        signal[mask] = replacement[mask]

        noisy[i, :length] = signal

    noisy = pd.DataFrame(noisy, columns=features.columns)
    return pd.concat([noisy, labels], axis=1)

# Scale and shift the heartbeat signal vertically (per-sample)
def add_amplitude_drift(df, valid_lengths, scale_range=(0.95, 1.05), shift_range=(-0.01, 0.01)):
    features = df.iloc[:, :-1].copy()
    labels = df.iloc[:, -1]

    drifted = features.to_numpy().copy()
    for i, length in enumerate(valid_lengths):
        scale = np.random.uniform(*scale_range)
        shift = np.random.uniform(*shift_range)
        drifted[i, :length] = features.iloc[i, :length].to_numpy() * scale + shift

    drifted = pd.DataFrame(drifted, columns=features.columns)
    return pd.concat([drifted, labels], axis=1)


# Shift the heartbeat signal horizontally (per-sample)
def add_signal_shift(df, valid_lengths, shift_range=(0, 0.5)):
    features = df.iloc[:, :-1].copy()
    labels = df.iloc[:, -1]

    shifted = features.to_numpy().copy()
    for i, length in enumerate(valid_lengths):
        signal = features.iloc[i, :length].to_numpy()
        shift = np.random.uniform(*shift_range)
        shift_len = int(length * shift)
        shifted_signal = np.roll(signal, shift_len)
        shifted[i, :length] = shifted_signal
        # Leave padding untouched

    shifted = pd.DataFrame(shifted, columns=features.columns)
    return pd.concat([shifted, labels], axis=1)

# Add min max and range to the dataset
def add_min_max(df, valid_lengths):
    features = df.iloc[:, :-1].copy()
    labels = df.iloc[:, -1]

    mins, maxs, ranges = [], [], []
    for i, length in enumerate(valid_lengths):
        signal = features.iloc[i, :length].to_numpy()
        mn, mx = signal.min(), signal.max()
        mins.append(mn)
        maxs.append(mx)
        ranges.append(mx - mn)

    features["min"] = mins
    features["max"] = maxs
    features["range"] = ranges

    return pd.concat([features, labels], axis=1)

# Add mean and std to the dataset
def add_mean_std(df, valid_lengths):
    features = df.iloc[:, :-1].copy()
    labels = df.iloc[:, -1]

    means, stds = [], []
    for i, length in enumerate(valid_lengths):
        signal = features.iloc[i, :length].to_numpy()
        means.append(signal.mean())
        stds.append(signal.std())

    features["mean"] = means
    features["std"] = stds

    return pd.concat([features, labels], axis=1)
