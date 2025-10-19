import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset
import time
from sklearn.model_selection import StratifiedKFold

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, loader, criterion):
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

def test_model(model, train_dl, test_dl, epochs = 10, validation = False):
    text = "val" if validation else "test"

    test_starttime = time.time()

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_test_acc, best_state = 0.0, None
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

        test_loss, test_acc = evaluate(model, test_dl, criterion)
        scheduler.step(test_loss)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

        epoch_endtime = time.time()
        time_taken = epoch_endtime - epoch_starttime

        # print(f"Epoch {ep:02d} | {text}_loss: {test_loss:.4f} | {text}_acc: {test_acc:.4f} | time_taken: {time_taken:.4f}s")

        test_endtime = time.time()
        time_taken = test_endtime - test_starttime
    print(f"Best {text}_acc: {best_test_acc:.4f} | time_taken: {time_taken:.4f}s")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_test_acc

def kFold_validation(model, x_train, y_train, epochs = 10, fold = 5):
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)

    init_parameters = {key: value.clone() for key,value in model.state_dict().items()}

    fold_accuracies = []
    for train_idx, val_idx in skf.split(x_train, y_train):

        train_x, train_y = x_train[train_idx].copy(), y_train[train_idx].copy()
        val_x, val_y = x_train[val_idx].copy(), y_train[val_idx].copy()

        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)

        train_x = train_x[:, None, :]
        val_x = val_x[:, None, :]

        train_fold_ds = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long())
        val_fold_ds = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).long())

        train_loader = DataLoader(train_fold_ds, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_fold_ds, batch_size=128, shuffle=False)

        model.load_state_dict(init_parameters) # Reset the model before every test
        model.to(device)

        fold_accuracies.append(test_model(model, train_loader, val_loader, epochs, validation=True)) # Test and append the accuracy

    return sum(fold_accuracies)/fold



