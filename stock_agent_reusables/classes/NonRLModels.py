import torch
from torch import nn

class LinearModel(nn.Module):
  def __init__(self, in_features, hidden_units, out_features):
    super().__init__()
    self.lin_layer = nn.Sequential(
        nn.Linear(
            in_features=in_features,
            out_features=hidden_units
        ),
        nn.ReLU(),
        nn.Linear(
            in_features=hidden_units,
            out_features=hidden_units
        ),
        nn.ReLU(),
        nn.Linear(
            in_features=hidden_units,
            out_features=out_features
        )
    )

  def forward(self, x):
    return self.lin_layer(x)

class LSTMModel(nn.Module):
  def __init__(self, in_features, hidden_units, out_features, stacked_layer_count, device):
    super().__init__()
    self.hidden_units = hidden_units
    self.stacked_layer_count = stacked_layer_count
    self.lstm = nn.LSTM(input_size=in_features,
                        hidden_size=hidden_units,
                        num_layers=stacked_layer_count,
                        batch_first=True)
    self.fc = nn.Linear(in_features=hidden_units,
                               out_features=out_features)
    self.device = device

  def forward(self, X):
    batch_size = len(X)
    h0 = torch.zeros(self.stacked_layer_count, batch_size, self.hidden_units)
    c0 = torch.zeros(self.stacked_layer_count, batch_size, self.hidden_units)
    out, _ = self.lstm(X, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out