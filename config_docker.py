# config_docker.py - Docker-optimized configuration
import os

CONFIG = {
    "seq_len": 100,
    "batch_size": 32,
    "hidden_size": 64,
    "num_layers": 1,
    "sensor_embed_dim": 4,
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "class_weights": [1.0, 5.0],
    "random_seed": 42,
    # Use environment variables or default paths for Docker
    "train_csv": os.getenv("TRAIN_CSV_PATH", "/app/data/train.csv"),
    "val_split": 0.2,
    "model_save_path": os.getenv("MODEL_SAVE_PATH", "/app/models/best_model.pt"),
    "weight_decay": 1e-5,
    "patience": 3,
    "lower": 0.01,
    "upper": 0.99,
    # Additional Docker-specific paths
    "sensor_encoder_path": "/app/models/sensor_encoder.pkl",
    "scaler_path": "/app/models/scaler.pkl",
    "output_dir": "/app/outputs"
}