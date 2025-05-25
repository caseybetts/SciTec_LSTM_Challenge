# Author: Casey Betts/ChatGPT
# Verifies the dataset is in a compatable format with the LSTM

import numpy as np
import matplotlib.pyplot as plt

# Verify the dataset is working
def verify_dataset(dataset, loader, windows, y_true, y_pred):
    # Check 2: Check Sample Values 
    for x_numeric, x_sensor, y in loader:
        print("x_numeric shape:", x_numeric.shape)  # (batch_size, seq_len, num_features)
        print("x_sensor shape:", x_sensor.shape)    # (batch_size, seq_len)
        print("y shape:", y.shape)                  # (batch_size, seq_len)
        break

    x_numeric, x_sensor, y = dataset[0]
    print("x_numeric[0]:", x_numeric[0])  # Should be a 4D vector
    print("x_sensor[0]:", x_sensor[0])    # Should be an integer
    print("y[0]:", y[0])                  # Should be a class index

    # Check 4: Visual Spot Check

    seq_index = 0  # First sequence in the batch

    true_seq = y_true[seq_index].cpu().numpy()
    pred_seq = y_pred[seq_index].cpu().numpy()


    plt.figure(figsize=(12, 4))
    plt.plot(true_seq, label='True Labels', marker='o', alpha=0.7)
    plt.plot(pred_seq, label='Predicted Labels', linestyle='--', marker='x', alpha=0.7)
    plt.xlabel("Time Step")
    plt.ylabel("Class Index")
    plt.title("Per-step Classification for One Sequence")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Check 6: Distribution Checks

    all_labels = np.concatenate([w["y"] for w in windows])
    unique, counts = np.unique(all_labels, return_counts=True)
    print("Label distribution:", dict(zip(unique, counts)))