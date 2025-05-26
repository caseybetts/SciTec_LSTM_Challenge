# inference.py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import LSTMTimeStepClassifier
from data import make_windows, TimeSeriesDataset
from config import CONFIG
import pickle
import os
import argparse

def load_encoder_scaler():
    # If you saved these during training, load them here.
    # Otherwise, fit them on the train set and save for future use.
    # For this example, we'll assume you saved them as .pkl files.
    with open("sensor_encoder.pkl", "rb") as f:
        sensor_encoder = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return sensor_encoder, scaler

def preprocess_test_data(test_csv, sensor_encoder, scaler):
    df = pd.read_csv(test_csv)
    df["sensor_idx"] = sensor_encoder.transform(df["sensor_id"])
    df[["latitude", "longitude", "altitude", "radiometric_intensity"]] = scaler.transform(
        df[["latitude", "longitude", "altitude", "radiometric_intensity"]]
    )
    return df

def create_test_windows(df, seq_len):
    windows = []
    for _, group in df.groupby("track_id"):
        windows.extend(make_windows(group, seq_len))
    return windows

def run_inference(input_csv, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoders/scalers
    sensor_encoder, scaler = load_encoder_scaler()

    # Load and preprocess test data
    df = preprocess_test_data(input_csv, sensor_encoder, scaler)
    windows = create_test_windows(df, CONFIG["seq_len"])
    test_dataset = TimeSeriesDataset(windows, CONFIG["seq_len"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])

    # Load model
    model = LSTMTimeStepClassifier(
        input_size=4,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        num_classes=2,
        sensor_vocab_size=len(sensor_encoder.classes_),
        sensor_embed_dim=CONFIG["sensor_embed_dim"],
        dropout=0.3
    ).to(device)
    model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=device))
    model.eval()

    # Run inference
    all_preds = []
    with torch.no_grad():
        for x_numeric, x_sensor, _ in test_loader:
            x_numeric, x_sensor = x_numeric.to(device), x_sensor.to(device)
            logits = model(x_numeric, x_sensor)
            preds = torch.argmax(logits, dim=-1)  # (batch, seq_len)
            all_preds.append(preds.cpu())

    # Concatenate predictions and flatten
    all_preds = torch.cat(all_preds, dim=0).numpy().flatten()

    # Map predictions back to original rows
    # (Assumes windows are in order and no overlap)
    # If you need to map back to original test.csv rows, you may need to save indices during windowing.

    # Save predictions to CSV
    pd.DataFrame({"prediction": all_preds}).to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSTM inference on a CSV file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="test_predictions.csv", help="Path to output CSV file")
    args = parser.parse_args()

    run_inference(args.input, args.output)