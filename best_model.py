import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
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

test_model(CNN(num_classes=classes), train_loader, test_loader, epochs=1, validation=False, list_data=True)

seeds = [10, 20, 30, 40, 50] # Add more seed to test models on different seed for averaging

'''
    Part 1: No augmentation and engineering
'''

models = {
    "MixModel1": [],
}

# MixModel
# for cnn in [16, 32, 64]:
#     for rnn in [16, 32, 64]:
#         for fc in [16, 32, 64]:
#             models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[cnn, cnn, cnn] ,h_rnn=[rnn, rnn], extra_features=extra_features, features_fc=[fc, fc]).to(device))

# for cnn in [128]:
#     for rnn in [128]:
#         for fc in [128]:
#             models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[cnn, cnn, cnn] ,h_rnn=[rnn, rnn], extra_features=extra_features, features_fc=[fc, fc]).to(device))

# for cnn in [64]:
#     for rnn in [64]:
#         for fc in [32]:
#             models["MixModel1"].append(MixModel1(num_classes=classes, h_cnn=[cnn, cnn, cnn] ,h_rnn=[rnn, rnn], extra_features=extra_features, features_fc=[fc, fc]).to(device))

print(models["MixModel1"][0])


# Write down each model details into the file
for key,value in models.items():
    for i, model in enumerate(value):
        log.write([f"{key} {i+1}", model], fileType="det")

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

        l_acc, l_loss, l_auc, l_time_taken, l_infer_time = [], [], [], [], []
        for seed in seeds: # Average over 5 runs
            set_seed(seed)

            model_1 = copy.deepcopy(model)

            metric, (accuracy, loss, auc_score, time_taken), inference_time = kFold_validation(model_1, x_train, y_train, epochs = global_epoch, fold = 5, by_AUC=by_AUC)

            total_metric += metric

            print(f"Training {key} model {i+1}, accuracy: {accuracy:.4f}, loss: {loss:.4f}, auc: {auc_score:.4f}, time_taken: {time_taken:.4f}, inference_time: {inference_time:.4f}")

            # Writing into raw file
            log.write([f"Training {key} model {i+1}, accuracy: {accuracy:.4f}, loss: {loss:.4f}, auc: {auc_score:.4f}, time_taken: {time_taken:.4f}"])
            log.bar()

            # Writing into complied file
            l_acc.append(accuracy)
            l_loss.append(loss)
            l_auc.append(auc_score)
            l_time_taken.append(time_taken)
            l_infer_time.append(inference_time)

        avg_acc = sum(l_acc) / len(l_acc)
        avg_loss = sum(l_loss) / len(l_loss)
        avg_auc = sum(l_auc) / len(l_auc)
        avg_time_taken = sum(l_time_taken) / len(l_time_taken) # Average per seed
        avg_infer_time = sum(l_infer_time) / len(l_infer_time)
        log.write([f"{key} {i+1}", "acc"] + l_acc + [avg_acc], fileType="comp")
        log.write([f"", "loss"] + l_loss + [avg_loss], fileType="comp")
        log.write([f"", "auc"] + l_auc + [avg_auc], fileType="comp")
        log.write([f"", "time_taken"] + l_time_taken + [avg_time_taken], fileType="comp")
        log.write([f"", "inference_time"] + l_infer_time + [avg_infer_time], fileType="comp")

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
