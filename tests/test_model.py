import torch
from model import LSTMTimeStepClassifier

def test_lstm_forward():
    model = LSTMTimeStepClassifier(
        input_size=4, hidden_size=8, num_layers=1, num_classes=2,
        sensor_vocab_size=5, sensor_embed_dim=2, dropout=0.1
    )
    x_numeric = torch.randn(2, 10, 4)  # batch=2, seq_len=10, features=4
    x_sensor = torch.randint(0, 5, (2, 10))
    logits = model(x_numeric, x_sensor)
    assert logits.shape == (2, 10, 2)
