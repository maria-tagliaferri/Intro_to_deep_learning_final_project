import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, hyperparameter_config):
        super(LSTMModel, self).__init__()

        self.input_size = hyperparameter_config['input_size']
        self.hidden_size = 50
        self.num_layers = 2
        self.output_size = hyperparameter_config['output_size']
        self.dropout = 0.2

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # x: (B, T, input_size)

        lstm_out, _ = self.lstm(x)
        # lstm_out: (B, T, hidden)

        last_timestep = lstm_out[:, -1, :]   # (B, hidden)

        out = self.fc(last_timestep)         # (B, output_size)

        return out