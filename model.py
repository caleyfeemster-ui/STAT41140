"""
model.py — LSTM model for weather prediction
STAT41140: AI for Weather and Climate

Architecture:
  Input covariates → LSTM layers → Fully connected output → prediction
"""

import torch
import torch.nn as nn


class WeatherLSTM(nn.Module):
    """
    LSTM network that takes a sequence of past weather observations
    and predicts the next value (e.g. tomorrow's rainfall).

    Args:
        input_size:   number of input features (covariates) per time step
        hidden_size:  number of units in each LSTM layer
        num_layers:   how many LSTM layers to stack
        output_size:  number of values to predict (1 for a single variable)
        dropout:      dropout rate between LSTM layers (regularisation)
    """

    def __init__(self, input_size=4, hidden_size=64, num_layers=2,
                 output_size=1, dropout=0.2):
        super(WeatherLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Core LSTM — processes the time sequence
        # batch_first=True means input shape is (batch, sequence, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully connected layer — maps final hidden state → prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: tensor of shape (batch_size, seq_len, input_size)

        Returns:
            out: tensor of shape (batch_size, output_size)
        """
        # Initialise hidden and cell states to zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Run through LSTM
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take only the output at the LAST time step
        # — this is the network's "summary" of the whole sequence
        last_step = lstm_out[:, -1, :]   # shape: (batch_size, hidden_size)

        # Map to prediction
        out = self.fc(last_step)          # shape: (batch_size, output_size)
        return out


if __name__ == "__main__":
    # Quick sanity check — run a dummy batch through the model
    model = WeatherLSTM(input_size=4, hidden_size=64, num_layers=2)
    dummy = torch.randn(8, 10, 4)   # batch=8, seq_len=10, features=4
    pred = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {pred.shape}")   # should be (8, 1)
    print("Model architecture:")
    print(model)
