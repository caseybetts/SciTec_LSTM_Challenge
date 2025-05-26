# model.py
import torch
import torch.nn as nn

class LSTMTimeStepClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sensor_vocab_size=None, sensor_embed_dim=0, dropout=0.3):
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
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, x_numeric, x_sensor=None):
        if self.use_sensor_embedding and x_sensor is not None:
            sensor_embed = self.sensor_embedding(x_sensor)
            x = torch.cat([x_numeric, sensor_embed], dim=-1)
        else:
            x = x_numeric
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        return logits