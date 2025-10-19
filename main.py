import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from models import CNN, RNN, LSTM, MixModel1
from datasets import get_ptbdb_dataset, get_mitbih_dataset
from datasets import data_eng, data_aug
from testing import kFold_validation, test_model
import copy

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


device = "cuda" if torch.cuda.is_available() else "cpu"

# Choose which dataset to use
datasets = ["mitbih", "ptbdb"]
dataset = datasets[1]

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

dataset_loader = {
    "mitbih": get_mitbih_dataset,
    "ptbdb": get_ptbdb_dataset
}

classes, x_train, y_train, x_test, y_test = dataset_loader[dataset](d_aug=d_aug, d_eng=d_eng) # Should be augmented and train/test split using 80/20

models = []

# CNN
models.append(CNN(num_classes=classes, h_cnn=[16, 16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))
# models.append(CNN(num_classes=classes, h_cnn=[8, 8, 8], extra_features=extra_features, features_fc=[8, 8]).to(device)) # appending different models with different sizes
# models.append(CNN(num_classes=classes, h_cnn=[16, 16, 16], extra_features=extra_features, features_fc=[4, 4]).to(device))

# RNN
models.append(RNN(num_classes=classes, h_rnn=[16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# LSTM
models.append(LSTM(num_classes=classes, h_rnn=[16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# MixModel
models.append(MixModel1(num_classes=classes, h_cnn=[16, 16, 16] ,h_rnn=[16, 16], extra_features=extra_features, features_fc=[32, 32]).to(device))

# Used for grid search
best_metric = 0
best_model_idx = None

by_AUC = False
metric = "AUC score" if by_AUC else "accuracy"

for i, model in enumerate(models):

    model_1 = copy.deepcopy(model)

    model_metric = kFold_validation(model_1, x_train, y_train, epochs = 10, fold = 5, by_AUC=by_AUC)
    print(f"Training model {i+1}, {metric}: {model_metric:.4f}")
    if model_metric > best_metric:
        best_metric = model_metric
        best_model_idx = i

best_model = models[best_model_idx] # Get the best model for retraining using train_ds and test_ds (based on accuracy)
print(f"Best model is {best_model_idx+1}")
print(best_model)

scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

scaled_x_train = scaled_x_train[:, None, :]
scaled_x_test = scaled_x_test[:, None, :]

train_fold_ds = TensorDataset(torch.from_numpy(scaled_x_train).float(), torch.from_numpy(y_train).long())
test_fold_ds = TensorDataset(torch.from_numpy(scaled_x_test).float(), torch.from_numpy(y_test).long())

train_loader = DataLoader(train_fold_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_fold_ds, batch_size=128, shuffle=False)

test_model(best_model, train_loader, test_loader, epochs=10, validation=False)

# Classification rate/ Confusion matrix
full_logits = []
full_labels = []
for x, y in test_loader:
    best_model.eval()
    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        logits = best_model(x)
        full_logits.append(logits.cpu())
        full_labels.append(y.cpu())

full_logits = torch.cat(full_logits, dim=0)
full_labels = torch.cat(full_labels, dim=0)
full_preds = torch.argmax(full_logits, dim=1)

y_true = full_labels.numpy()
y_pred = full_preds.numpy()

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Classification report (precision, recall, F1)
report = classification_report(y_true, y_pred)
print("Classification Report:\n", report)


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