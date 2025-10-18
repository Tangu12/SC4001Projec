import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=2, h_cnn=[32, 64, 128]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, h_cnn[0], kernel_size=7, padding=3, stride=1), nn.BatchNorm1d(h_cnn[0]), nn.ReLU(),
            nn.Conv1d(h_cnn[0], h_cnn[1], kernel_size=5, padding=2, stride=2), nn.BatchNorm1d(h_cnn[1]), nn.ReLU(),  # 187 -> ~94
            nn.Conv1d(h_cnn[1], h_cnn[2], kernel_size=3, padding=1, stride=2), nn.BatchNorm1d(h_cnn[2]), nn.ReLU(), # 94 -> ~47
            nn.MaxPool1d(2, ceil_mode=True),  # -> (128,24)
            nn.AdaptiveAvgPool1d(1)  # -> (B, 128, 1)
        )
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.net(x).squeeze(-1)  # (B,128,1)->(B,128)
        return self.fc(x)

class RNN(nn.Module):
    def __init__(self, num_classes=2, final = False, h_rnn = [4,16]):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=1, hidden_size=h_rnn[0], num_layers=1, batch_first=True)
        self.rnn2 = nn.RNN(input_size=h_rnn[0], hidden_size=h_rnn[1], num_layers=1, batch_first=True)
        self.fc = nn.Linear(h_rnn[1], num_classes)
        self.final = final
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.fc(x)
        if self.final:
            return x[:,-1,:]
        return x

class LSTM(nn.Module):
    def __init__(self, num_classes=2, final = False, h_rnn = [4,16]):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=h_rnn[0], num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=h_rnn[0], hidden_size=h_rnn[1], num_layers=1, batch_first=True)
        self.fc = nn.Linear(h_rnn[1], num_classes)
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
    def __init__(self, num_classes=2, final = False, h_cnn=[32, 64, 128], h_rnn=[4,16] ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, h_cnn[0], kernel_size=7, padding=3, stride=1), nn.BatchNorm1d(h_cnn[0]), nn.ReLU(),
            nn.Conv1d(h_cnn[0], h_cnn[1], kernel_size=5, padding=2, stride=2), nn.BatchNorm1d(h_cnn[1]), nn.ReLU(),  # 187 -> ~94
            nn.Conv1d(h_cnn[1], h_cnn[2], kernel_size=3, padding=1, stride=2), nn.BatchNorm1d(h_cnn[2]), nn.ReLU(), # 94 -> ~47
            nn.MaxPool1d(2, ceil_mode=True),  # -> (128,24)
        )
        self.lstm1 = nn.LSTM(input_size=h_cnn[2], hidden_size=h_rnn[0], num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=h_rnn[0], hidden_size=h_rnn[1], num_layers=1, batch_first=True)
        self.fc = nn.Linear(h_rnn[1], num_classes) #[RNN output, extra features]
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