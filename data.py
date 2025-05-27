# data.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_and_preprocess_data(config):
    logger.info(f"Loading data from {config['train_csv']}")
    df = pd.read_csv(config["train_csv"])
    sensor_encoder = LabelEncoder()
    df["sensor_idx"] = sensor_encoder.fit_transform(df["sensor_id"])

    numeric_cols = ["latitude", "longitude", "altitude", "radiometric_intensity"]

    # Outlier clipping
    for col in numeric_cols:
        low = df[col].quantile(config["lower"])
        high = df[col].quantile(config["upper"])
        df[col] = df[col].clip(low, high)

    # Smoothing
    df = smooth_features(df, numeric_cols, window=3)

    # Scaling
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save encoder and scaler for inference
    with open("sensor_encoder.pkl", "wb") as f:
        pickle.dump(sensor_encoder, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Data loaded and preprocessed.")
    return df, sensor_encoder, scaler

def smooth_features(df, columns, window=3):
    df = df.sort_values(['track_id', 'timestamp'])
    for col in columns:
        df[col] = df.groupby('track_id')[col].transform(lambda x: x.rolling(window, min_periods=1, center=True).mean())
    return df

def make_windows(group_df, seq_len):
    numeric_feats = group_df[["latitude", "longitude", "altitude", "radiometric_intensity"]].to_numpy()
    sensor_ids = group_df["sensor_idx"].to_numpy()
    # Only get labels if present
    if "reentry_phase" in group_df.columns:
        labels = group_df["reentry_phase"].to_numpy()
    else:
        labels = np.zeros(len(group_df), dtype=int)  # or None, or np.full(..., -1)
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

def create_windows(df, seq_len):
    windows = []
    for _, group in df.groupby("track_id"):
        windows.extend(make_windows(group, seq_len))
    return windows

class TimeSeriesDataset(Dataset):
    def __init__(self, windows, seq_len):
        self.windows = windows
        self.seq_len = seq_len

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = self.windows[idx]
        x_numeric = torch.tensor(item["x_numeric"], dtype=torch.float32)
        x_sensor = torch.tensor(item["x_sensor"], dtype=torch.long)
        y = torch.tensor(item["y"], dtype=torch.long)
        assert y.shape == (self.seq_len,), "Each sequence should have per-step labels"
        return x_numeric, x_sensor, y

def get_datasets(windows, seq_len, val_split, seed):
    from torch.utils.data import DataLoader
    train_windows, val_windows = train_test_split(windows, test_size=val_split, random_state=seed)
    train_dataset = TimeSeriesDataset(train_windows, seq_len)
    val_dataset = TimeSeriesDataset(val_windows, seq_len)
    return train_dataset, val_dataset