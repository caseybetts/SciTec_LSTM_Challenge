import torch
from model import LSTMTimeStepClassifier

def test_training_step():
    model = LSTMTimeStepClassifier(
        input_size=4, hidden_size=8, num_layers=1, num_classes=2,
        sensor_vocab_size=5, sensor_embed_dim=2, dropout=0.1
    )
    x_numeric = torch.randn(2, 10, 4)
    x_sensor = torch.randint(0, 5, (2, 10))
    y = torch.randint(0, 2, (2, 10))
    loss_fn = torch.nn.CrossEntropyLoss()
    logits = model(x_numeric, x_sensor)
    loss = loss_fn(logits.view(-1, 2), y.view(-1))
    loss.backward()
    assert loss.item() > 0
