from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

# Scalers

def get_scaler_for_data(data: List[int]):
  scaler = MinMaxScaler()
  data = np.array(data).reshape(-1, 1)
  scaler.fit(data)
  return scaler

def apply_scaler_to_data(data: List[int], scaler: MinMaxScaler, inverse=False):
  data = np.array(data).reshape(-1, 1)
  if inverse:
    data = scaler.inverse_transform(data).reshape(-1)
  else:
    data = scaler.transform(data).reshape(-1)
  return data

# Data transforming

class StockDataset(Dataset):
  def __init__(self, data, labels):
    self.data = data
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    input = torch.tensor(self.data[index], dtype=torch.float32)
    label = torch.tensor(self.labels[index], dtype=torch.long)
    return input, label

def get_data_loaders(data, labels, train_split_percentage=0.8, batch_size=32):
  train_end_idx = int(len(data) * train_split_percentage)
  X_train, y_train = data[:train_end_idx], labels[:train_end_idx]
  X_test, y_test = data[train_end_idx:], labels[train_end_idx:]
  train_dataset, test_dataset = StockDataset(X_train, y_train), StockDataset(X_test, y_test)

  return DataLoader(dataset=train_dataset, batch_size=batch_size), DataLoader(dataset=test_dataset, batch_size=batch_size)

def lookback_data_transform(data: pd.DataFrame, lookback_interval: int, lstm_transform=False):
  new_data = []
  if lstm_transform:
    new_data.append(data.iloc[:lookback_interval, :].values)
    for i in range(lookback_interval+1, len(data)):
      new_data.append(np.append(new_data[i-lookback_interval-1][1:], [data.iloc[i-lookback_interval][:].values], axis=0))
  else:
    for i in range(lookback_interval, len(data)):
      new_data.append(data.iloc[i-lookback_interval:i, :].values.flatten())
    
  return new_data

"""
Contains functions for training and testing a PyTorch model.
Retrieved from Daniel Bourke's PyTorch course.
"""
from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
  """
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item()

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval()

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)

          # 1. Forward pass
          test_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          # Calculate and accumulate accuracy
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
    For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
  """
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # Return the filled results at the end of the epochs
  return results

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_model_predictions(stock_prices: List[float], signals: List[int]):
  # Map signals to colors and labels
  color_mapping = {0: 'green', 1: 'red', 2: 'orange'}

  # Plot stock prices with connecting lines
  plt.plot(stock_prices, label='Stock Prices', linestyle='-', color='grey')

  # Plot signals with corresponding colors
  for i, signal in enumerate(signals):
      plt.scatter(i, stock_prices[i], color=color_mapping[signal])

  # Create a custom legend with specific markers
  legend_handles = [
      mpatches.Circle((0, 0), 0.1, color='green', label='Buy'),
      mpatches.Circle((0, 0), 0.1, color='red', label='Sell'),
      mpatches.Circle((0, 0), 0.1, color='orange', label='Hold')
  ]

  # Add legend with custom handles
  plt.legend(handles=legend_handles)

  # Add labels and title
  plt.xlabel('Time')
  plt.ylabel('Stock Price')
  plt.title('Stock Prices with Buy/Sell/Hold Signals')

  # Show the plot
  plt.show()