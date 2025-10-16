import pandas as pd
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

df_normal = pd.read_csv("ECG_heartbeats/ptbdb_abnormal.csv", header=None)
df_abnormal = pd.read_csv("ECG_heartbeats/ptbdb_abnormal.csv", header=None)
# print(df_normal.shape)
# print(df_abnormal.shape)
df = pd.concat([df_normal, df_abnormal], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# I don't really und what is gg on here :P
X = df.iloc[:, :-1].values.astype(np.float32)  # selecting every column except the last column
y = df.iloc[:, -1].values.astype(np.int64)     # just selecting the last column

# Standardize per-feature (helps training)
m, s = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True) + 1e-8
X = (X - m) / s

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
class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, stride=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2), nn.BatchNorm1d(64), nn.ReLU(),  # 187 -> ~94
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2), nn.BatchNorm1d(128), nn.ReLU(), # 94 -> ~47
            nn.MaxPool1d(2),  # -> ~23
            nn.AdaptiveAvgPool1d(1)  # -> (B, 128, 1)
        )
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.net(x).squeeze(-1)  # (B,128,1)->(B,128)
        return self.fc(x)

model = CNN(num_classes=2).to(device)

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
epochs = 15
for ep in range(1, epochs + 1):
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

    print(f"Epoch {ep:02d} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

# Restore best
if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Best val_acc: {best_val_acc:.4f}")