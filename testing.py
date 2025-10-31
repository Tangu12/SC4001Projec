import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch.nn.functional as F
import time
from sklearn.model_selection import StratifiedKFold

torch.manual_seed(42)

def set_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = "cuda" if torch.cuda.is_available() else "cpu"

# def evaluate(model, loader, criterion):
#     model.eval()
#     total, correct, running_loss = 0, 0, 0.0
#     with torch.no_grad():
#         for xb, yb in loader:
#             xb, yb = xb.to(device), yb.to(device)
#             logits = model(xb)
#             loss = criterion(logits, yb)
#             running_loss += loss.item() * yb.size(0)
#             pred = logits.argmax(1)
#             correct += (pred == yb).sum().item()
#             total += yb.size(0)
#     return running_loss / total, correct / total

def evaluate(model, loader, criterion, return_probs=False):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            running_loss += loss.item() * yb.size(0)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

            if return_probs:
                probs = F.softmax(logits, dim=1)  # convert logits to probabilities
                all_outputs.append(probs.cpu())
            else:
                all_outputs.append(logits.cpu())

            all_labels.append(yb.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return running_loss / total, correct / total, all_outputs, all_labels

def test_model(model, train_dl, test_dl, epochs = 10, validation = False, by_AUC = False, list_data = False):
    text = "val" if validation else "test"
    metric = "AUC score" if by_AUC else "accuracy"

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_test_metric, best_state, metrics = 0.0, None, []
    data_dict = {
        "acc": [],
        "loss": [],
        "auc": [],
        "time": []
    }
    for ep in range(1, epochs + 1):
        epoch_starttime = time.time()

        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        test_loss, test_acc, output, labels = evaluate(model, test_dl, criterion, return_probs=True)
        scheduler.step(test_loss)

        epoch_endtime = time.time()
        time_taken = epoch_endtime - epoch_starttime

        if output.shape[1] == 2:  # binary classification
            auc_score = roc_auc_score(labels.numpy(), output[:, 1].numpy())
        else:  # multi-class
            auc_score = roc_auc_score(labels.numpy(), output.numpy(), multi_class="ovr")

        if ((not by_AUC) and (test_acc > best_test_metric)) or (by_AUC and (auc_score > best_test_metric)):
            best_test_metric = auc_score if by_AUC else test_acc
            metrics = [test_acc, test_loss, auc_score]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        data_dict["acc"].append(test_acc)
        data_dict["loss"].append(test_loss)
        data_dict["auc"].append(auc_score)
        data_dict["time"].append(time_taken)
        # print(f"Epoch {ep:02d} | {text}_loss: {test_loss:.4f} | {text}_acc: {test_acc:.4f} | time_taken: {time_taken:.4f}s")

    total_time_taken = sum(data_dict["time"])
    # print(f"Best {text} {metric}: {best_test_metric:.4f} | time_taken: {total_time_taken:.4f}s")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Inference time
    model.eval()
    data = iter(test_dl)
    xb, yb = next(data)
    xb, yb = xb.to(device), yb.to(device)
    with torch.no_grad():
        start_time = time.time()
        outputs = model(xb)
        end_time = time.time()
    inference_time = (end_time - start_time)/xb.size(0)

    metrics.append(total_time_taken) # We are returning the best metrics + the total time taken to run all epochs

    if list_data:
        return data_dict

    return best_test_metric, metrics, inference_time

def kFold_validation(model, x_train, y_train, epochs = 10, fold = 5, by_AUC = False):
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)

    init_parameters = {key: value.clone() for key,value in model.state_dict().items()}

    test_metric = 0
    fold_metrics = [0, 0, 0, 0]
    inference_time = 0
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

        best_test_metric, new_metric, inference_time = test_model(model, train_loader, val_loader, epochs, validation=True, by_AUC=by_AUC)

        test_metric += best_test_metric # Adds whatever metric is used for comparing
        fold_metrics = [x + y for x,y in zip(fold_metrics, new_metric)] # Adds the accuracy, loss, AUC_score of the best_model (chosen accuracy if by_AUC is false otherwise by AUC score)
        inference_time += inference_time # Adds up all the inference time to be averaged

    return test_metric/fold, [i/fold for i in fold_metrics], inference_time/fold # Returns accuracy, loss, AUC_score averaged over the k folds



