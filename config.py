# config.py
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
    "train_csv": "data/train.csv",
    "val_split": 0.2,
    "model_save_path": "best_model.pt",
    "weight_decay":1e-5,
    "patience":3,
    "lower":0.01,
    "upper":0.99
}