import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from models import CNN, RNN, LSTM, MixModel1
from datasets import get_ptbdb_dataset, get_mitbih_dataset
from datasets import data_eng, data_aug
from testing import kFold_validation, test_model, set_seed
from logger import log
import copy
import time

time_start = time.time()


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = "cuda" if torch.cuda.is_available() else "cpu"



# Choose which dataset to use
datasets = ["mitbih", "ptbdb"]
dataset = datasets[1]

# # Choose the data augmentation
# d_aug = []
# d_aug.append(data_aug.gaussian_noise)
# d_aug.append(data_aug.salt_pepper)
# d_aug.append(data_aug.signal_shift)
# d_aug.append(data_aug.amplitude_drift)
#
# # Choose the data engineering
# d_eng = []
# d_eng.append(data_eng.min_max)
# d_eng.append(data_eng.mean_std)
# d_eng.append(data_eng.skew)
# d_eng.append(data_eng.kurtosis)
#
# # Calculate the extra features
# count = {
#     data_eng.min_max: 3,
#     data_eng.mean_std: 2,
#     data_eng.skew: 1,
#     data_eng.kurtosis: 1,
# }
# extra_features = sum(count[i] for i in d_eng)

dataset_loader = {
    "mitbih": get_mitbih_dataset,
    "ptbdb": get_ptbdb_dataset
}

classes, x_train, y_train, x_test, y_test = dataset_loader[dataset]() # Should be augmented and train/test split using 80/20

log.write([f"The dataset is {dataset}",
           f"No data augmentation applied",
           f"No data engineering applied",
           ], fileType="all")

# Test run
scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

scaled_x_train = scaled_x_train[:, None, :]
scaled_x_test = scaled_x_test[:, None, :]

train_fold_ds = TensorDataset(torch.from_numpy(scaled_x_train).float(), torch.from_numpy(y_train).long())
test_fold_ds = TensorDataset(torch.from_numpy(scaled_x_test).float(), torch.from_numpy(y_test).long())

train_loader = DataLoader(train_fold_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_fold_ds, batch_size=128, shuffle=False)

test_model(CNN(), train_loader, test_loader, epochs=1, validation=False, list_data=True)

seeds = [10, 20, 30, 40, 50] # Add more seed to test models on different seed for averaging

'''
    Part 1: No augmentation and engineering
'''

models = {
    "CNN": [],
    "RNN": [],
    "LSTM": [],
    "MixModel1": [],
}

# CNN
models["CNN"].append(CNN(num_classes=classes, h_cnn=[8, 8, 8]).to(device))
models["CNN"].append(CNN(num_classes=classes, h_cnn=[16, 16, 16]).to(device))
models["CNN"].append(CNN(num_classes=classes, h_cnn=[32, 32, 32]).to(device))
# models["CNN"].append(CNN(num_classes=classes, h_cnn=[16, 16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# RNN
models["RNN"].append(RNN(num_classes=classes, h_rnn=[8, 8]).to(device))
models["RNN"].append(RNN(num_classes=classes, h_rnn=[16, 16]).to(device))
models["RNN"].append(RNN(num_classes=classes, h_rnn=[32, 32]).to(device))
# models["RNN"].append(RNN(num_classes=classes, h_rnn=[16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# LSTM
models["LSTM"].append(LSTM(num_classes=classes, h_rnn=[8, 8]).to(device))
models["LSTM"].append(LSTM(num_classes=classes, h_rnn=[16, 16]).to(device))
models["LSTM"].append(LSTM(num_classes=classes, h_rnn=[32, 32]).to(device))
# models["LSTM"].append(LSTM(num_classes=classes, h_rnn=[16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# MixModel
models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[8, 8, 8] ,h_rnn=[8, 8]).to(device))
models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[16, 16, 16] ,h_rnn=[16, 16]).to(device))
models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[32, 32, 32] ,h_rnn=[32, 32]).to(device))
# models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[16, 16, 16] ,h_rnn=[16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# Write down each model details into the file
for key,value in models.items():
    for i, model in enumerate(value):
        log.write([f"{key} {i}", model], fileType="det")

# Used for grid search
best_metric = 0
best_model = None
global_epoch = 10

by_AUC = False

log.write(["Model", "Metric", "Run 1", "Run 2", "Run 3", "Run 4", "Run 5", "Avg"], fileType="comp")

for key,value in models.items():

    for i, model in enumerate(value):

        total_metric = 0

        log.write([f"{key} {i}", model], fileType="raw")

        l_acc, l_loss, l_auc, l_time_taken = [], [], [], []
        for seed in seeds: # Average over 5 runs
            set_seed(seed)

            model_1 = copy.deepcopy(model)

            metric, (accuracy, loss, auc_score, time_taken) = kFold_validation(model_1, x_train, y_train, epochs = global_epoch, fold = 5, by_AUC=by_AUC)

            total_metric += metric

            print(f"Training {key} model {i+1}, accuracy: {accuracy:.4f}, loss: {loss:.4f}, auc: {auc_score:.4f}, time_taken: {time_taken:.4f}")

            # Writing into raw file
            log.write([f"Training {key} model {i+1}, accuracy: {accuracy:.4f}, loss: {loss:.4f}, auc: {auc_score:.4f}, time_taken: {time_taken:.4f}"])
            log.bar()

            # Writing into complied file
            l_acc.append(accuracy)
            l_loss.append(loss)
            l_auc.append(auc_score)
            l_time_taken.append(time_taken)

        avg_acc = sum(l_acc) / len(l_acc)
        avg_loss = sum(l_loss) / len(l_loss)
        avg_auc = sum(l_auc) / len(l_auc)
        avg_time_taken = sum(l_time_taken) / len(l_time_taken) # Average per seed
        log.write([f"{key} {i}", "acc"] + l_acc + [avg_acc], fileType="comp")
        log.write([f"", "loss"] + l_loss + [avg_loss], fileType="comp")
        log.write([f"", "auc"] + l_auc + [avg_auc], fileType="comp")
        log.write([f"", "time_taken"] + l_time_taken + [avg_time_taken], fileType="comp")

        if total_metric/5 > best_metric:
            best_metric = total_metric/5
            best_model = [key, i]

model_choice = models[best_model[0]][best_model[1]] # Get the best model for retraining using train_ds and test_ds (based on accuracy)
print(f"Best model is {best_model[0]} model {best_model[1]+1}")
print(model_choice)

log.write([f"Best model is {best_model[0]} model {best_model[1]+1}"])
log.write([model_choice, "Testing on train/test set"])

scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

scaled_x_train = scaled_x_train[:, None, :]
scaled_x_test = scaled_x_test[:, None, :]

train_fold_ds = TensorDataset(torch.from_numpy(scaled_x_train).float(), torch.from_numpy(y_train).long())
test_fold_ds = TensorDataset(torch.from_numpy(scaled_x_test).float(), torch.from_numpy(y_test).long())

train_loader = DataLoader(train_fold_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_fold_ds, batch_size=128, shuffle=False)

data_dict = test_model(model_choice, train_loader, test_loader, epochs=global_epoch, validation=False, list_data=True)

log.write([str(data_dict["acc"]), str(data_dict["loss"]), str(data_dict["auc"]), str(data_dict["time"])])

log.write(["Model", "Metric"] + [f"Epoch {i+1}" for i in range(global_epoch)], fileType="comp")
log.write([f"{best_model[0]} {best_model[1]+1}", "acc"] + data_dict["acc"], fileType="comp")
log.write([f"", "loss"] + data_dict["loss"], fileType="comp")
log.write([f"", "auc"] + data_dict["auc"], fileType="comp")
log.write([f"", "time_taken"] + data_dict["time"], fileType="comp")

'''
    Part 2: With augmentation and no engineering
'''

# Choose the data augmentation
d_aug = []
d_aug.append(data_aug.gaussian_noise)
d_aug.append(data_aug.salt_pepper)
d_aug.append(data_aug.signal_shift)
d_aug.append(data_aug.amplitude_drift)

classes, x_train, y_train, x_test, y_test = dataset_loader[dataset](d_aug = d_aug) # Should be augmented and train/test split using 80/20

log.write([f"The dataset is {dataset}",
           f"Data augment: {d_aug}",
           f"No data engineering applied",
           ], fileType="all")

models = {
    "CNN": [],
    "RNN": [],
    "LSTM": [],
    "MixModel1": [],
}

# CNN
models["CNN"].append(CNN(num_classes=classes, h_cnn=[8, 8, 8]).to(device))
models["CNN"].append(CNN(num_classes=classes, h_cnn=[16, 16, 16]).to(device))
models["CNN"].append(CNN(num_classes=classes, h_cnn=[32, 32, 32]).to(device))
# models["CNN"].append(CNN(num_classes=classes, h_cnn=[16, 16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# RNN
models["RNN"].append(RNN(num_classes=classes, h_rnn=[8, 8]).to(device))
models["RNN"].append(RNN(num_classes=classes, h_rnn=[16, 16]).to(device))
models["RNN"].append(RNN(num_classes=classes, h_rnn=[32, 32]).to(device))
# models["RNN"].append(RNN(num_classes=classes, h_rnn=[16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# LSTM
models["LSTM"].append(LSTM(num_classes=classes, h_rnn=[8, 8]).to(device))
models["LSTM"].append(LSTM(num_classes=classes, h_rnn=[16, 16]).to(device))
models["LSTM"].append(LSTM(num_classes=classes, h_rnn=[32, 32]).to(device))
# models["LSTM"].append(LSTM(num_classes=classes, h_rnn=[16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# MixModel
models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[8, 8, 8] ,h_rnn=[8, 8]).to(device))
models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[16, 16, 16] ,h_rnn=[16, 16]).to(device))
models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[32, 32, 32] ,h_rnn=[32, 32]).to(device))
# models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[16, 16, 16] ,h_rnn=[16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# Write down each model details into the file
for key,value in models.items():
    for i, model in enumerate(value):
        log.write([f"{key} {i}", model], fileType="det")

# Used for grid search
best_metric = 0
best_model = None
global_epoch = 10

by_AUC = False

log.write(["Model", "Metric", "Run 1", "Run 2", "Run 3", "Run 4", "Run 5", "Avg"], fileType="comp")

for key,value in models.items():

    for i, model in enumerate(value):

        total_metric = 0

        log.write([f"{key} {i}", model], fileType="raw")

        l_acc, l_loss, l_auc, l_time_taken = [], [], [], []
        for seed in seeds: # Average over 5 runs
            set_seed(seed)

            model_1 = copy.deepcopy(model)

            metric, (accuracy, loss, auc_score, time_taken) = kFold_validation(model_1, x_train, y_train, epochs = global_epoch, fold = 5, by_AUC=by_AUC)

            total_metric += metric

            print(f"Training {key} model {i+1}, accuracy: {accuracy:.4f}, loss: {loss:.4f}, auc: {auc_score:.4f}, time_taken: {time_taken:.4f}")

            # Writing into raw file
            log.write([f"Training {key} model {i+1}, accuracy: {accuracy:.4f}, loss: {loss:.4f}, auc: {auc_score:.4f}, time_taken: {time_taken:.4f}"])
            log.bar()

            # Writing into complied file
            l_acc.append(accuracy)
            l_loss.append(loss)
            l_auc.append(auc_score)
            l_time_taken.append(time_taken)

        avg_acc = sum(l_acc) / len(l_acc)
        avg_loss = sum(l_loss) / len(l_loss)
        avg_auc = sum(l_auc) / len(l_auc)
        avg_time_taken = sum(l_time_taken) / len(l_time_taken) # Average per seed
        log.write([f"{key} {i}", "acc"] + l_acc + [avg_acc], fileType="comp")
        log.write([f"", "loss"] + l_loss + [avg_loss], fileType="comp")
        log.write([f"", "auc"] + l_auc + [avg_auc], fileType="comp")
        log.write([f"", "time_taken"] + l_time_taken + [avg_time_taken], fileType="comp")

        if total_metric/5 > best_metric:
            best_metric = total_metric/5
            best_model = [key, i]

model_choice = models[best_model[0]][best_model[1]] # Get the best model for retraining using train_ds and test_ds (based on accuracy)
print(f"Best model is {best_model[0]} model {best_model[1]+1}")
print(model_choice)

log.write([f"Best model is {best_model[0]} model {best_model[1]+1}"])
log.write([model_choice, "Testing on train/test set"])

scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

scaled_x_train = scaled_x_train[:, None, :]
scaled_x_test = scaled_x_test[:, None, :]

train_fold_ds = TensorDataset(torch.from_numpy(scaled_x_train).float(), torch.from_numpy(y_train).long())
test_fold_ds = TensorDataset(torch.from_numpy(scaled_x_test).float(), torch.from_numpy(y_test).long())

train_loader = DataLoader(train_fold_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_fold_ds, batch_size=128, shuffle=False)

data_dict = test_model(model_choice, train_loader, test_loader, epochs=global_epoch, validation=False, list_data=True)

log.write([str(data_dict["acc"]), str(data_dict["loss"]), str(data_dict["auc"]), str(data_dict["time"])])

log.write(["Model", "Metric"] + [f"Epoch {i+1}" for i in range(global_epoch)], fileType="comp")
log.write([f"{best_model[0]} {best_model[1]+1}", "acc"] + data_dict["acc"], fileType="comp")
log.write([f"", "loss"] + data_dict["loss"], fileType="comp")
log.write([f"", "auc"] + data_dict["auc"], fileType="comp")
log.write([f"", "time_taken"] + data_dict["time"], fileType="comp")

'''
    Part 3: With augmentation and engineering
'''

# Choose the data augmentation
d_aug = []
d_aug.append(data_aug.gaussian_noise)
d_aug.append(data_aug.salt_pepper)
d_aug.append(data_aug.signal_shift)
d_aug.append(data_aug.amplitude_drift)

# Choose the data engineering
d_eng = []
d_eng.append(data_eng.min_max)
d_eng.append(data_eng.mean_std)
d_eng.append(data_eng.skew)
d_eng.append(data_eng.kurtosis)

# Calculate the extra features
count = {
    data_eng.min_max: 3,
    data_eng.mean_std: 2,
    data_eng.skew: 1,
    data_eng.kurtosis: 1,
}
extra_features = sum(count[i] for i in d_eng)

classes, x_train, y_train, x_test, y_test = dataset_loader[dataset](d_aug = d_aug, d_eng = d_eng) # Should be augmented and train/test split using 80/20

log.write([f"The dataset is {dataset}",
           f"Data augment: {d_aug}",
           f"Data engineering: {d_eng}",
           ], fileType="all")

models = {
    "CNN": [],
    "RNN": [],
    "LSTM": [],
    "MixModel1": [],
}

# CNN
models["CNN"].append(CNN(num_classes=classes, h_cnn=[8, 8, 8], extra_features=extra_features, features_fc=[8, 8]).to(device))
models["CNN"].append(CNN(num_classes=classes, h_cnn=[16, 16, 16], extra_features=extra_features, features_fc=[16, 16]).to(device))
models["CNN"].append(CNN(num_classes=classes, h_cnn=[32, 32, 32], extra_features=extra_features, features_fc=[32, 32]).to(device))
# models["CNN"].append(CNN(num_classes=classes, h_cnn=[16, 16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# RNN
models["RNN"].append(RNN(num_classes=classes, h_rnn=[8, 8], extra_features=extra_features, features_fc=[8, 8]).to(device))
models["RNN"].append(RNN(num_classes=classes, h_rnn=[16, 16], extra_features=extra_features, features_fc=[16, 16]).to(device))
models["RNN"].append(RNN(num_classes=classes, h_rnn=[32, 32], extra_features=extra_features, features_fc=[32, 32]).to(device))
# models["RNN"].append(RNN(num_classes=classes, h_rnn=[16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# LSTM
models["LSTM"].append(LSTM(num_classes=classes, h_rnn=[8, 8], extra_features=extra_features, features_fc=[8, 8]).to(device))
models["LSTM"].append(LSTM(num_classes=classes, h_rnn=[16, 16], extra_features=extra_features, features_fc=[16, 16]).to(device))
models["LSTM"].append(LSTM(num_classes=classes, h_rnn=[32, 32], extra_features=extra_features, features_fc=[32, 32]).to(device))
# models["LSTM"].append(LSTM(num_classes=classes, h_rnn=[16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# MixModel
models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[8, 8, 8] ,h_rnn=[8, 8], extra_features=extra_features, features_fc=[8, 8]).to(device))
models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[16, 16, 16] ,h_rnn=[16, 16], extra_features=extra_features, features_fc=[16, 16]).to(device))
models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[32, 32, 32] ,h_rnn=[32, 32], extra_features=extra_features, features_fc=[32, 32]).to(device))
# models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[16, 16, 16] ,h_rnn=[16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# Write down each model details into the file
for key,value in models.items():
    for i, model in enumerate(value):
        log.write([f"{key} {i}", model], fileType="det")

# Used for grid search
best_metric = 0
best_model = None
global_epoch = 10

by_AUC = False

log.write(["Model", "Metric", "Run 1", "Run 2", "Run 3", "Run 4", "Run 5", "Avg"], fileType="comp")

for key,value in models.items():

    for i, model in enumerate(value):

        total_metric = 0

        log.write([f"{key} {i+1}", model], fileType="raw")

        l_acc, l_loss, l_auc, l_time_taken = [], [], [], []
        for seed in seeds: # Average over 5 runs
            set_seed(seed)

            model_1 = copy.deepcopy(model)

            metric, (accuracy, loss, auc_score, time_taken) = kFold_validation(model_1, x_train, y_train, epochs = global_epoch, fold = 5, by_AUC=by_AUC)

            total_metric += metric

            print(f"Training {key} model {i+1}, accuracy: {accuracy:.4f}, loss: {loss:.4f}, auc: {auc_score:.4f}, time_taken: {time_taken:.4f}")

            # Writing into raw file
            log.write([f"Training {key} model {i+1}, accuracy: {accuracy:.4f}, loss: {loss:.4f}, auc: {auc_score:.4f}, time_taken: {time_taken:.4f}"])
            log.bar()

            # Writing into complied file
            l_acc.append(accuracy)
            l_loss.append(loss)
            l_auc.append(auc_score)
            l_time_taken.append(time_taken)

        avg_acc = sum(l_acc) / len(l_acc)
        avg_loss = sum(l_loss) / len(l_loss)
        avg_auc = sum(l_auc) / len(l_auc)
        avg_time_taken = sum(l_time_taken) / len(l_time_taken) # Average per seed
        log.write([f"{key} {i}", "acc"] + l_acc + [avg_acc], fileType="comp")
        log.write([f"", "loss"] + l_loss + [avg_loss], fileType="comp")
        log.write([f"", "auc"] + l_auc + [avg_auc], fileType="comp")
        log.write([f"", "time_taken"] + l_time_taken + [avg_time_taken], fileType="comp")

        if total_metric/5 > best_metric:
            best_metric = total_metric/5
            best_model = [key, i]

model_choice = models[best_model[0]][best_model[1]] # Get the best model for retraining using train_ds and test_ds (based on accuracy)
print(f"Best model is {best_model[0]} model {best_model[1]+1}")
print(model_choice)

log.write([f"Best model is {best_model[0]} model {best_model[1]+1}"])
log.write([model_choice, "Testing on train/test set"])

scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

scaled_x_train = scaled_x_train[:, None, :]
scaled_x_test = scaled_x_test[:, None, :]

train_fold_ds = TensorDataset(torch.from_numpy(scaled_x_train).float(), torch.from_numpy(y_train).long())
test_fold_ds = TensorDataset(torch.from_numpy(scaled_x_test).float(), torch.from_numpy(y_test).long())

train_loader = DataLoader(train_fold_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_fold_ds, batch_size=128, shuffle=False)

data_dict = test_model(model_choice, train_loader, test_loader, epochs=global_epoch, validation=False, list_data=True)

log.write([str(data_dict["acc"]), str(data_dict["loss"]), str(data_dict["auc"]), str(data_dict["time"])])

log.write(["Model", "Metric"] + [f"Epoch {i+1}" for i in range(global_epoch)], fileType="comp")
log.write([f"{best_model[0]} {best_model[1]+1}", "acc"] + data_dict["acc"], fileType="comp")
log.write([f"", "loss"] + data_dict["loss"], fileType="comp")
log.write([f"", "auc"] + data_dict["auc"], fileType="comp")
log.write([f"", "time_taken"] + data_dict["time"], fileType="comp")

time_end = time.time()

print(f"Time taken: {((time_end - time_start)/3600)%60:.2f} hrs {((time_end - time_start)/60)%60:.2f} mins {(time_end - time_start)%60:.2f} seconds")

#
# # Classification rate/ Confusion matrix
# full_logits = []
# full_labels = []
# for x, y in test_loader:
#     best_model.eval()
#     with torch.no_grad():
#         x, y = x.to(device), y.to(device)
#         logits = best_model(x)
#         full_logits.append(logits.cpu())
#         full_labels.append(y.cpu())
#
# full_logits = torch.cat(full_logits, dim=0)
# full_labels = torch.cat(full_labels, dim=0)
# full_preds = torch.argmax(full_logits, dim=1)
#
# y_true = full_labels.numpy()
# y_pred = full_preds.numpy()
#
# # Accuracy
# acc = accuracy_score(y_true, y_pred)
# print(f"Accuracy: {acc:.3f}")
#
# # Confusion Matrix
# cm = confusion_matrix(y_true, y_pred)
# print("Confusion Matrix:\n", cm)
#
# # Classification report (precision, recall, F1)
# report = classification_report(y_true, y_pred)
# print("Classification Report:\n", report)


# TODO
# Testing metrics to find the best hyperparameter per model
# (Loss, Accuracy, classification rate, multi-class AUC) against each set of hyperparameter
# Confusion matrix

# Modify model so that it also takes in statistical features (to be combined with RNN output before FC) -DONE
# Potential multi-scale model - split sequence into N sub-sequences ->

# Without data
# Best val accuracy: 0.8089 | time_taken: 2.9021s
# Best val accuracy: 0.7393 | time_taken: 2.3497s
# Best val accuracy: 0.7650 | time_taken: 2.2872s
# Best val accuracy: 0.7599 | time_taken: 2.2755s
# Best val accuracy: 0.7839 | time_taken: 2.2706s
# Training model 1, accuracy: 0.7714
# Best val accuracy: 0.8811 | time_taken: 2.3971s
# Best val accuracy: 0.8750 | time_taken: 2.3840s
# Best val accuracy: 0.8711 | time_taken: 2.2609s
# Best val accuracy: 0.8750 | time_taken: 2.4828s
# Best val accuracy: 0.8776 | time_taken: 2.4916s
# Training model 2, accuracy: 0.8760
# Best val accuracy: 0.9012 | time_taken: 3.0777s
# Best val accuracy: 0.9089 | time_taken: 3.0470s
# Best val accuracy: 0.8956 | time_taken: 3.2068s
# Best val accuracy: 0.8960 | time_taken: 3.0586s
# Best val accuracy: 0.8930 | time_taken: 2.8497s
# Training model 3, accuracy: 0.8990

# With Data
# Best val accuracy: 0.8424 | time_taken: 2.8331s
# Best val accuracy: 0.8402 | time_taken: 2.2187s
# Best val accuracy: 0.8252 | time_taken: 2.2843s
# Best val accuracy: 0.8385 | time_taken: 2.1425s
# Best val accuracy: 0.8449 | time_taken: 2.1098s
# Training model 1, accuracy: 0.8382
# Best val accuracy: 0.8785 | time_taken: 2.2905s
# Best val accuracy: 0.8647 | time_taken: 2.3213s
# Best val accuracy: 0.8630 | time_taken: 2.4602s
# Best val accuracy: 0.8660 | time_taken: 2.3764s
# Best val accuracy: 0.8707 | time_taken: 2.3557s
# Training model 2, accuracy: 0.8686
# Best val accuracy: 0.9047 | time_taken: 2.6552s
# Best val accuracy: 0.8978 | time_taken: 2.7048s
# Best val accuracy: 0.8896 | time_taken: 2.6717s
# Best val accuracy: 0.8943 | time_taken: 2.7245s
# Best val accuracy: 0.8960 | time_taken: 2.8226s
# Training model 3, accuracy: 0.8965