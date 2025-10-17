import pandas as pd
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import neurokit2 as nk
import matplotlib.pyplot as plt
import time


torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Choose which dataset to use
datasets = ["mitbih", "ptbdb"]
dataset = datasets[0]

train_dl, valid_dl, test_dl = None, None, None
classes = None

if dataset == "ptbdb":
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

elif dataset == "mitbih":
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

# data_iter = iter(train_dl)
#
# xs = [i+1 for i in range(187)]
#
#
# # Get the first batch
# x_batch, y_batch = next(data_iter)
# # print(x_batch[0].shape, y_batch.shape)
# # print(x_batch[0].numpy().shape)
# print(x_batch[14].numpy()[0])
#
#
# plt.plot(xs,x_batch[110].numpy()[0] )
# plt.show()
#
# # cleaned = nk.ecg_clean(x_batch[100].numpy()[0], sampling_rate=125 )
# #
# # # 2. Detect R-peaks
# # signals, info = nk.ecg_peaks(cleaned, sampling_rate=125)
# #
# # # 3. Compute P, Q, R, S, T peaks
# # peaks = nk.ecg_findpeaks(cleaned, sampling_rate=125)
# #
# # # 4. Compute interval features (HR, RR, QT, PR)
# # features = nk.ecg_intervalrelated(peaks, sampling_rate=125)
# #
# # print(features)



class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, stride=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2), nn.BatchNorm1d(64), nn.ReLU(),  # 187 -> ~94
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2), nn.BatchNorm1d(128), nn.ReLU(), # 94 -> ~47
            nn.MaxPool1d(2, ceil_mode=True),  # -> (128,24)
            nn.AdaptiveAvgPool1d(1)  # -> (B, 128, 1)
        )
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.net(x).squeeze(-1)  # (B,128,1)->(B,128)
        return self.fc(x)

class RNN(nn.Module):
    def __init__(self, num_classes=2, final = False):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=1, hidden_size=4, num_layers=1, batch_first=True)
        # self.rnn2 = nn.RNN(input_size=4, hidden_size=16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(4, num_classes)
        self.final = final
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.rnn1(x)
        # x, _ = self.rnn2(x)
        x = self.fc(x)
        if self.final:
            return x[:,-1,:]
        return x

class LSTM(nn.Module):
    def __init__(self, num_classes=2, final = False):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
        self.final = final
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        if self.final:
            return x[:,-1,:]
        return x

class MixModel1(nn.Module):
    def __init__(self, num_classes=2, final = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, stride=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2), nn.BatchNorm1d(64), nn.ReLU(),  # 187 -> ~94
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2), nn.BatchNorm1d(128), nn.ReLU(), # 94 -> ~47
            nn.MaxPool1d(2, ceil_mode=True),  # -> (128,24)
        )
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, num_classes) #[RNN output, extra features]
        self.final = final
    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        if self.final:
            return x[:,-1,:]
        return x

class MixModel2(nn.Module):
    def __init__(self, num_classes=2, final=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, stride=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True),  # -> (128,24)
        )
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, num_classes)
        self.final = final

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        if self.final:
            return x[:, -1, :]
        return x


def test_model(model):

    test_starttime = time.time()

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Training and Validation
    def evaluate(model, loader):
        model.eval()
        total, correct, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                running_loss += loss.item() * yb.size(0)
                pred = logits.argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return running_loss / total, correct / total

    best_val_acc, best_state = 0.0, None
    epochs = 10
    for ep in range(1, epochs + 1):
        epoch_starttime = time.time()

        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model, val_dl)
        scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

        epoch_endtime = time.time()
        time_taken = epoch_endtime - epoch_starttime

        print(f"Epoch {ep:02d} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | time_taken: {time_taken:.4f}s")

        test_endtime = time.time()
        time_taken = test_endtime - test_starttime
    print(f"Best val_acc: {best_val_acc:.4f} | time_taken: {time_taken:.4f}s")
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)


# model = CNN(num_classes= classes).to(device)
# test_model(model)

# model = RNN(num_classes=classes, final= True).to(device)
# test_model(model)

# model = LSTM(num_classes=classes, final= True).to(device)
# test_model(model)

# model = MixModel1(num_classes=classes, final=True).to(device)
# test_model(model)

# model = MixModel2(num_classes=classes, final=True).to(device)
# test_model(model)