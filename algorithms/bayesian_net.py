import torch
import torch.nn as nn


class BayesianNet(nn.Module):
    def __init__(self, window_size: int, dropout: float):
        super(BayesianNet, self).__init__()
        self.window_size = window_size
        self.dropout = dropout
        self.fc1 = nn.Linear(window_size, 128)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


class BayesianNetRanked(nn.Module):
    def __init__(self, window_size: int, dropout: float):
        super(BayesianNetRanked, self).__init__()
        self.window_size = window_size
        self.dropout = dropout
        self.fc1 = nn.Linear(window_size, 128)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class BayesianCNN(nn.Module):
    def __init__(self, window_size: int, dropout: float):
        super(BayesianCNN, self).__init__()

        self.window_size = window_size
        self.dropout_rate = dropout

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1,
                               padding=2)  # Assuming input is 1D and you retain the size
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)

        # Fully connected layers
        # The size of the linear layers might need adjustments based on the preceding layer sizes and chosen parameters.
        reduced_window_size = self.window_size  # Because of padding=2, window size doesn't reduce
        self.fc1 = nn.Linear(reduced_window_size * 64, 128)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension

        # Convolution operations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Flatten before passing to fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected operations
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))

        return torch.sigmoid(self.fc3(x))


class BayesianLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_layers: int):
        super(BayesianLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # LSTM layer expects input of shape (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)

        # Only take the output from the final timestep
        x = lstm_out[:, -1, :]

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))

        return torch.sigmoid(self.fc3(x))
