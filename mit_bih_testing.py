import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from models import CNN, RNN, LSTM, MixModel1
from datasets import get_ptbdb_dataset, get_mitbih_dataset
from datasets import data_eng, data_aug
from testing import kFold_validation, test_model, set_seed
from logger import log
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Choose which dataset to use
datasets = ["mitbih", "ptbdb"]
dataset = datasets[0]

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

classes, x_train, y_train, x_test, y_test = dataset_loader[dataset](d_aug= d_aug, d_eng=d_eng) # Should be augmented and train/test split using 80/20

log.write([f"The dataset is {dataset}",
           f"Testing with augmentation and engineering",
           f"Using MixModel1 as the best architecture",
           ], fileType="all")

global_epoch = 10

cnn, rnn, fc = 64, 64, 32
model_choice = MixModel1(num_classes=classes, h_cnn=[cnn, cnn, cnn] ,h_rnn=[rnn, rnn], extra_features=extra_features, features_fc=[fc, fc]).to(device)

scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

scaled_x_train = scaled_x_train[:, None, :]
scaled_x_test = scaled_x_test[:, None, :]

train_fold_ds = TensorDataset(torch.from_numpy(scaled_x_train).float(), torch.from_numpy(y_train).long())
test_fold_ds = TensorDataset(torch.from_numpy(scaled_x_test).float(), torch.from_numpy(y_test).long())

train_loader = DataLoader(train_fold_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_fold_ds, batch_size=128, shuffle=False)

time_start = time.time()

data_dict = test_model(model_choice, train_loader, test_loader, epochs=global_epoch, validation=False, list_data=True)

training_time = time.time() - time_start
print(training_time)

log.write([str(data_dict["acc"]), str(data_dict["loss"]), str(data_dict["auc"]), str(data_dict["time"])])

log.write(["Model", "Metric"] + [f"Epoch {i+1}" for i in range(global_epoch)], fileType="comp")
log.write([f"Chosen Optimal model", "acc"] + data_dict["acc"], fileType="comp")
log.write([f"", "loss"] + data_dict["loss"], fileType="comp")
log.write([f"", "auc"] + data_dict["auc"], fileType="comp")
log.write([f"", "time_taken"] + data_dict["time"], fileType="comp")

# Classification rate/ Confusion matrix
full_logits = []
full_labels = []
for x, y in test_loader:
    model_choice.eval()
    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        logits = model_choice(x)
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
