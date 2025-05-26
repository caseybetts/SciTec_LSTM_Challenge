# train.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from config import CONFIG
from data import load_and_preprocess_data, create_windows, get_datasets
from model import LSTMTimeStepClassifier
from utils import set_seed

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

def train():
    set_seed(CONFIG["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, sensor_encoder, scaler = load_and_preprocess_data(CONFIG)
    windows = create_windows(df, CONFIG["seq_len"])
    train_dataset, val_dataset = get_datasets(windows, CONFIG["seq_len"], CONFIG["val_split"], CONFIG["random_seed"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    model = LSTMTimeStepClassifier(
        input_size=4,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        num_classes=2,
        sensor_vocab_size=len(sensor_encoder.classes_),
        sensor_embed_dim=CONFIG["sensor_embed_dim"],
        dropout=0.3
    ).to(device)
    class_weights = torch.tensor(CONFIG["class_weights"]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    best_f1 = 0
    epochs_not_improved = 0
    for epoch in range(CONFIG["num_epochs"]):
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
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} - Loss: {avg_loss:.4f}")
        _, val_f1 = evaluate(model, val_loader, device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print("Best model saved.")
            epochs_not_improved = 0
        else:
            epochs_not_improved += 1
            print(f"No improvment for {epochs_not_improved} epoch(s).")
            if epochs_not_improved >= CONFIG["patience"]:
                print("Early stopping triggered.")
                break

    print("Training complete.")

if __name__ == "__main__":
    train()