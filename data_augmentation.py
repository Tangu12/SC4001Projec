import numpy as np

# At each time stamp, add random noise using error ~ N(0,sigma)
def add_gaussian_noise(signal, sigma=0.01):
    noise = np.random.normal(0, sigma, signal.shape)
    return signal + noise

# At each time stamp, have a probability of being the max/min value of the signal
def add_salt_pepper(signal, prob=0.01):
    noisy = signal.copy()
    mask = np.random.rand(*signal.shape) < prob
    noisy[mask] = np.random.choice([signal.min(), signal.max()], size=mask.sum())
    return noisy

# Scale and shift the heartbeat signal vertically
def add_amplitude_drift(signal, scale_range=(0.95, 1.05), shift_range=(-0.01, 0.01)):
    scale = np.random.uniform(*scale_range)
    shift = np.random.uniform(*shift_range)
    return signal * scale + shift

# Shift the heartbeat signal horizontally
def add_signal_shift(signal, shift_range=(0, 0.5)):
    shift = np.random.uniform(*shift_range)
    shift_len = int(len(signal) * shift)
    return np.roll(signal, shift_len)

