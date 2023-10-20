import torch
import torch.nn as nn


class BayesianNet(nn.Module):
    def __init__(self, window_size: int, dropout: float, temperature: float):
        super(BayesianNet, self).__init__()
        self.window_size = window_size
        self.dropout = dropout
        self.temperature = temperature

        self.fc1 = nn.Linear(window_size, 256)  # Increased units
        self.bn1 = nn.BatchNorm1d(256)  # Batch normalization
        self.dropout1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(p=dropout)

        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        return torch.sigmoid(self.fc4(x) / self.temperature)



class BayesianCNN(nn.Module):
    def __init__(self, window_size: int, dropout: float, temperature: float):
        super(BayesianCNN, self).__init__()

        self.window_size = window_size
        self.dropout_rate = dropout
        self.temperature = temperature

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)

        # Fully connected layers
        reduced_window_size = self.window_size  # Because of padding=2, window size doesn't reduce
        self.fc1 = nn.Linear(reduced_window_size * 64, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch normalization
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)  # Batch normalization
        self.dropout2 = nn.Dropout(p=self.dropout_rate)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)  # Batch normalization
        self.dropout3 = nn.Dropout(p=self.dropout_rate)

        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension

        # Convolution operations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Flatten before passing to fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected operations
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        return torch.sigmoid(self.fc4(x) / self.temperature)

class BayesianLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_layers: int, temperature: float):
        super(BayesianLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temperature = temperature

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 256)  # Adjusted to 256 units to match BayesianNet
        self.bn1 = nn.BatchNorm1d(256)  # Added BatchNorm
        self.dropout1 = nn.Dropout(p=dropout)  # Renamed to dropout1 for consistency

        self.fc2 = nn.Linear(256, 128)  # Adjusted to 128 units to match BayesianNet
        self.bn2 = nn.BatchNorm1d(128)  # Added BatchNorm
        self.dropout2 = nn.Dropout(p=dropout)  # Renamed to dropout2 for consistency

        self.fc3 = nn.Linear(128, 64)  # Adjusted to 64 units to match BayesianNet
        self.bn3 = nn.BatchNorm1d(64)  # Added BatchNorm
        self.dropout3 = nn.Dropout(p=dropout)  # Renamed to dropout3 for consistency

        self.fc4 = nn.Linear(64, 1)  # Added an additional fc layer to match BayesianNet

    def forward(self, x):
        # LSTM layer expects input of shape (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)

        # Only take the output from the final timestep
        x = lstm_out[:, -1, :]

        # Fully connected layers with Batch Normalization and Dropout
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        return torch.sigmoid(self.fc4(x) / self.temperature)
