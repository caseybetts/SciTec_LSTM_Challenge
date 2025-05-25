# Author: ChatGPT/Casey Betts
# Date: 2025-05-23
# This script prepares a CSV for use in LSTM model, creates the LSTM and evaluates the accuracy of the model 

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Data_Verification import verify_dataset


# Read the CSV file
df = pd.read_csv(r"mlops-coding-challenge-EXTERNAL\mlops-coding-challenge-EXTERNAL\data\train.csv")

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
seq_len = 100

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

model = LSTMTimeStepClassifier(
    input_size=4,                # num numerical features
    hidden_size=64,
    num_layers=1,
    num_classes=2,               # reentry_phase classes
    sensor_vocab_size=20,
    sensor_embed_dim=4
)

model.eval()
with torch.no_grad():
    x_numeric, x_sensor, y_true = next(iter(loader))  # Get a single batch
    logits = model(x_numeric, x_sensor)               # Forward pass
    y_pred = torch.argmax(logits, dim=-1)             # Predicted classes

# (Optional) Checks to ensure dataset is compatable with LSTM
# verify_dataset(dataset, loader, windows, y_true, y_pred)

# Define loss and optimizer
class_weights = torch.tensor([1.0, 5.0])
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define model trainer
def train(model, loader, loss_fn, optimizer, num_epochs=2):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in loader:
            x_numeric, x_sensor, y = batch

            optimizer.zero_grad()

            logits = model(x_numeric, x_sensor)  # [B, T, C]
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))  # reshape

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        # print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

# Train LSTM
train(model, loader, loss_fn, optimizer, num_epochs=10)

# Define SequenceDataset
class SequenceDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        x_numeric = torch.tensor(window["x_numeric"], dtype=torch.float32)   # shape: [T, num_features]
        x_sensor  = torch.tensor(window["x_sensor"], dtype=torch.long)       # shape: [T]
        y         = torch.tensor(window["y"], dtype=torch.long)              # shape: [T]
        return x_numeric, x_sensor, y


# Random but reproducible split
train_windows, val_windows = train_test_split(windows, test_size=0.2, random_state=42)

train_dataset = SequenceDataset(train_windows)
val_dataset = SequenceDataset(val_windows)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Define evaluation function
def evaluate(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            x_numeric, x_sensor, y = batch

            logits = model(x_numeric, x_sensor)
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).view(-1).numpy()
    all_labels = torch.cat(all_labels).view(-1).numpy()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')  # Use 'macro' for >2 classes

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    print(classification_report(all_labels, all_preds, digits=4))

    return acc, f1

# Run evaluation
evaluate(model, val_loader)













