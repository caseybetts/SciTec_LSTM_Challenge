# train.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from config import CONFIG
from data import load_and_preprocess_data, create_windows, get_datasets
from model import LSTMTimeStepClassifier
from utils import set_seed
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_numeric, x_sensor, y in val_loader:
            x_numeric, x_sensor, y = x_numeric.to(device), x_sensor.to(device), y.to(device)
            logits = model(x_numeric, x_sensor)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds).view(-1).numpy()
    all_labels = torch.cat(all_labels).view(-1).numpy()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    print(classification_report(all_labels, all_preds, digits=4))
    return acc, f1

def baseline_reentry_phase(x_numeric_batch):
    """
    x_numeric_batch: (batch, seq_len, 4) numpy array
    Returns: (batch, seq_len) numpy array of baseline predictions
    """
    # Altitude is the 3rd column (index 2)
    altitudes = x_numeric_batch[:, :, 2]
    batch_size, seq_len = altitudes.shape
    preds = np.zeros((batch_size, seq_len), dtype=int)
    for i in range(batch_size):
        decreasing_count = 0
        in_reentry = False
        for t in range(1, seq_len):
            if altitudes[i, t] < altitudes[i, t-1]:
                decreasing_count += 1
            else:
                decreasing_count = 0
            if decreasing_count >= 30:
                in_reentry = True
            if in_reentry:
                preds[i, t] = 1
    return preds

def evaluate_baseline(val_loader, device):
    all_preds = []
    all_labels = []
    for x_numeric, x_sensor, y in val_loader:
        # x_numeric: (batch, seq_len, 4)
        # y: (batch, seq_len)
        x_numeric_np = x_numeric.numpy()
        baseline_preds = baseline_reentry_phase(x_numeric_np)
        all_preds.append(baseline_preds.reshape(-1))
        all_labels.append(y.numpy().reshape(-1))
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    logger.info(f"Baseline Validation Accuracy: {acc:.4f}")
    logger.info(f"Baseline Validation F1 Score: {f1:.4f}")
    logger.info("\n" + classification_report(all_labels, all_preds, digits=4))
    return acc, f1

def train():
    logger.info("Starting training process.")
    set_seed(CONFIG["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading and preprocessing data.")
    df, sensor_encoder, scaler = load_and_preprocess_data(CONFIG)
    windows = create_windows(df, CONFIG["seq_len"])
    train_dataset, val_dataset = get_datasets(windows, CONFIG["seq_len"], CONFIG["val_split"], CONFIG["random_seed"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

    # Evaluate baseline
    logger.info("Evaluating baseline classifier on validation set...")
    evaluate_baseline(val_loader, device)

    model = LSTMTimeStepClassifier(
        input_size=4,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        num_classes=2,
        sensor_vocab_size=len(sensor_encoder.classes_),
        sensor_embed_dim=CONFIG["sensor_embed_dim"],
        dropout=0.3
    ).to(device)
    logger.info("Model instantiated.")
    class_weights = torch.tensor(CONFIG["class_weights"]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    best_f1 = 0
    epochs_not_improved = 0
    for epoch in range(CONFIG["num_epochs"]):
        logger.info(f"Epoch {epoch+1} started.")
        model.train()
        total_loss = 0.0
        for x_numeric, x_sensor, y in train_loader:
            x_numeric, x_sensor, y = x_numeric.to(device), x_sensor.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_numeric, x_sensor)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} completed. Loss: {avg_loss:.4f}")
        _, val_f1 = evaluate(model, val_loader, device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            logger.info("Best model saved.")
            epochs_not_improved = 0
        else:
            epochs_not_improved += 1
            logger.info(f"No improvement for {epochs_not_improved} epoch(s).")
            if epochs_not_improved >= CONFIG["patience"]:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete.")

if __name__ == "__main__":
    train()