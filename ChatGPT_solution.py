# Author: ChatGPT/Casey Betts
# Date: 2025-05-23
# This script prepares a CSV for use in LSTM model training

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import Dataset

# Read the CSV file
df = pd.read_csv(r"mlops-coding-challenge-EXTERNAL\mlops-coding-challenge-EXTERNAL\data\train.csv")
# df["timestamp"] = pd.to_datetime(df["timestamp"])
# df = df.sort_values(["track_id", "timestamp"])


# Turn sensor_id into an integer index (for embedding):
sensor_encoder = LabelEncoder()
df["sensor_idx"] = sensor_encoder.fit_transform(df["sensor_id"])
sensor_vocab_size = len(sensor_encoder.classes_)  # used in model init

# Normalize Numeric Features
scaler = StandardScaler()
df[["latitude", "longitude", "altitude", "radiometric_intensity"]] = scaler.fit_transform(
    df[["latitude", "longitude", "altitude", "radiometric_intensity"]]
)

# Window the sequence data:

seq_len = 1000

def make_windows(group_df, seq_len):
    numeric_feats = group_df[["latitude", "longitude", "altitude", "radiometric_intensity"]].to_numpy()
    sensor_ids = group_df["sensor_idx"].to_numpy()
    labels = group_df["reentry_phase"].to_numpy()

    windows = []
    for start in range(0, len(group_df), seq_len):
        end = start + seq_len
        if end > len(group_df): break
        window = {
            "x_numeric": numeric_feats[start:end],
            "x_sensor": sensor_ids[start:end],
            "y": labels[start:end],
        }
        windows.append(window)
    return windows

windows = []
for _, group in df.groupby("track_id"):
    windows.extend(make_windows(group, seq_len))



# Convert to PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, windows, seq_len):
        self.windows = windows
        self.seq_len = seq_len

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = self.windows[idx]
        x_numeric = torch.tensor(item["x_numeric"], dtype=torch.float32)    # (seq_len, num_features)
        x_sensor = torch.tensor(item["x_sensor"], dtype=torch.long)         # (seq_len,)
        y = torch.tensor(item["y"], dtype=torch.long)                       # (seq_len,)
        assert y.shape == (self.seq_len,), "Each sequence should have per-step labels"

        return x_numeric, x_sensor, y

# Wrap in DataLoader
dataset = TimeSeriesDataset(windows, seq_len)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


# Step 3 PyTorch Model Code

import torch.nn as nn

class LSTMTimeStepClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sensor_vocab_size=None, sensor_embed_dim=0):
        super().__init__()

        self.use_sensor_embedding = sensor_vocab_size is not None and sensor_embed_dim > 0

        if self.use_sensor_embedding:
            self.sensor_embedding = nn.Embedding(sensor_vocab_size, sensor_embed_dim)
            self.lstm_input_size = input_size + sensor_embed_dim
        else:
            self.lstm_input_size = input_size

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x_numeric, x_sensor=None):
        """
        x_numeric: Tensor of shape (batch, seq_len, num_features)
        x_sensor:  LongTensor of shape (batch, seq_len) — optional sensor IDs
        """
        if self.use_sensor_embedding and x_sensor is not None:
            sensor_embed = self.sensor_embedding(x_sensor)  # (batch, seq_len, embed_dim)
            x = torch.cat([x_numeric, sensor_embed], dim=-1)
        else:
            x = x_numeric

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        logits = self.classifier(lstm_out)  # (batch, seq_len, num_classes)
        return logits

# Verify the dataset is working
# Check 3 Try Forward Pass

model = LSTMTimeStepClassifier(
    input_size=4,                # num numerical features
    hidden_size=64,
    num_layers=1,
    num_classes=3,               # reentry_phase classes
    sensor_vocab_size=20,
    sensor_embed_dim=4
)

model.eval()
with torch.no_grad():
    x_numeric, x_sensor, y_true = next(iter(loader))  # Get a single batch
    logits = model(x_numeric, x_sensor)               # Forward pass
    y_pred = torch.argmax(logits, dim=-1)             # Predicted classes


# Verify the dataset is working
def verify_dataset(dataset, loader, windows):
    # Check 2: Check Sample Values 
    for x_numeric, x_sensor, y in loader:
        print("x_numeric shape:", x_numeric.shape)  # (batch_size, seq_len, num_features)
        print("x_sensor shape:", x_sensor.shape)    # (batch_size, seq_len)
        print("y shape:", y.shape)                  # (batch_size, seq_len)
        break

    x_numeric, x_sensor, y = dataset[0]
    print("x_numeric[0]:", x_numeric[0])  # Should be a 4D vector
    print("x_sensor[0]:", x_sensor[0])    # Should be an integer
    print("y[0]:", y[0])                  # Should be a class index

    # Check 4: Visual Spot Check
    import matplotlib.pyplot as plt

    seq_index = 0  # First sequence in the batch

    true_seq = y_true[seq_index].cpu().numpy()
    pred_seq = y_pred[seq_index].cpu().numpy()


    plt.figure(figsize=(12, 4))
    plt.plot(true_seq, label='True Labels', marker='o', alpha=0.7)
    plt.plot(pred_seq, label='Predicted Labels', linestyle='--', marker='x', alpha=0.7)
    plt.xlabel("Time Step")
    plt.ylabel("Class Index")
    plt.title("Per-step Classification for One Sequence")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Check 6: Distribution Checks

    all_labels = np.concatenate([w["y"] for w in windows])
    unique, counts = np.unique(all_labels, return_counts=True)
    print("Label distribution:", dict(zip(unique, counts)))
# verify_dataset(dataset, loader, windows)

import torch.optim as optim

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()

for x_numeric, x_sensor, y in train_loader:
    optimizer.zero_grad()
    logits = model(x_numeric, x_sensor)  # [batch_size, seq_len, num_classes]

    loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))  # reshape
    loss.backward()
    optimizer.step()









